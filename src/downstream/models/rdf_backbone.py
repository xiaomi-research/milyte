#
# Copyright (C) 2026 Xiaomi Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing permissions and limitations under the License.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
import json
import math
import os
import random
from pathlib import Path

import yaml

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:
    Chem = None
    AllChem = None

DEFAULT_COMPONENT_SMILES = {

    "EC": "O=C1OCCO1",
    "EMC": "COC(=O)OCC",
    "DMC": "COC(=O)OC",
    "DEC": "CCOC(=O)OCC",
    "PC": "C1(=O)OC(C)CO1",
    "MA": "COC(=O)C",
    "EA": "CCOC(=O)C",
    "EP": "CCC(=O)OCC",
    "PP": "CCCC(=O)OCCC",
    "VN": "N#CCCCC",
    "LiPF6": "[Li+].F[P-](F)(F)(F)(F)F",
    "LiFSI": "[Li+].O=S(=O)([N-]S(=O)(=O)F)F",
}

def _gen_conformer_heavy_atoms(smiles: str, seed: int = 0, max_i_iters: int = 500):

    if Chem is None or AllChem is None:
        raise RuntimeError("RDKit is not installed or failed to import; please install rdkit.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        raise RuntimeError(f"RDKit EmbedMolecule failed: smiles={smiles}")

    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=int(max_i_iters))
    except Exception:

        pass

    conf = mol.GetConformer()
    coords_all = []
    z_all = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        z = atom.GetAtomicNum()

        if z > 1:
            coords_all.append([float(pos.x), float(pos.y), float(pos.z)])
            z_all.append(int(z))

    coords = np.asarray(coords_all, dtype=np.float32)
    atom_z = np.asarray(z_all, dtype=np.int64)

    if coords.shape[0] == 0:
        raise RuntimeError(f"No heavy atoms in generation result: smiles={smiles}")

    return coords, atom_z

def _get_component_name_from_mol_column(mol_column: str) -> str:

    if mol_column.startswith("mol_count_"):
        return mol_column.replace("mol_count_", "")
    return mol_column

def build_component_graphs_from_rdkit(
    mol_columns,
    component_smiles_map=None,
    cutoff: float = 6.0,
    seed: int = 0,
):

    component_smiles_map = component_smiles_map or DEFAULT_COMPONENT_SMILES

    component_graphs = []
    for i, mol_col in enumerate(mol_columns):
        comp = _get_component_name_from_mol_column(mol_col).strip()
        smiles = component_smiles_map.get(comp, None)
        if smiles is None:
            raise KeyError(f"Missing SMILES definition for component {comp} (add it to DEFAULT_COMPONENT_SMILES or config)")

        try:
            coords, atom_z = _gen_conformer_heavy_atoms(smiles, seed=seed + i)

            coords = coords - coords.mean(axis=0, keepdims=True)
        except Exception as e:

            print(f"Warning: RDKit 3D generation failed for component {comp}, using single-atom placeholder: {e}")

            fallback_z = 3 if comp.startswith("Li") else 6
            coords = np.zeros((1, 3), dtype=np.float32)
            atom_z = np.asarray([fallback_z], dtype=np.int64)

        Ni = coords.shape[0]
        if Ni <= 1:
            edge_src = np.zeros((0,), dtype=np.int64)
            edge_dst = np.zeros((0,), dtype=np.int64)
        else:
            dmat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            src_list = []
            dst_list = []
            for a in range(Ni):
                for b in range(Ni):
                    if a == b:
                        continue
                    if dmat[a, b] <= cutoff:
                        src_list.append(a)
                        dst_list.append(b)
            edge_src = np.asarray(src_list, dtype=np.int64)
            edge_dst = np.asarray(dst_list, dtype=np.int64)

        component_graphs.append({
            "coords": coords,
            "atom_z": atom_z,
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "component": comp,
        })

    return component_graphs

def _random_rotation_matrix(batch_size: int, device: torch.device) -> torch.Tensor:

    u1 = torch.rand(batch_size, device=device)
    u2 = torch.rand(batch_size, device=device)
    u3 = torch.rand(batch_size, device=device)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * math.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * math.pi * u3)

    R = torch.empty(batch_size, 3, 3, device=device, dtype=torch.float32)
    x, y, z, w = q1, q2, q3, q4

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def _rotate_vecs(R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

    if x.dim() < 3 or x.shape[-1] != 3:
        raise ValueError("x must have last dim 3")
    B = x.shape[0]
    x_flat = x.reshape(B, -1, 3)
    x_rot_flat = torch.einsum("bij,bmj->bmi", R, x_flat)
    return x_rot_flat.reshape_as(x)

class ComponentFeatureEncoder(nn.Module):

    def __init__(self, num_components, hidden_dim, output_dim, dropout=0.15):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(num_components, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, mol_fractions):
        return self.encoder(mol_fractions)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x, positions):
        batch_size, seq_len = positions.shape
        pos_encoding = self.pe[:, positions.reshape(-1)].reshape(batch_size, seq_len, -1)
        return x + pos_encoding

class RDFTransformerBlock(nn.Module):

    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.15):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )
    
    def forward(self, x, mask=None):
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        predictions = self.regression_head(encoded)
        
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            E_Pair = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            E_Pair = encoded.mean(dim=1)
        
        return predictions, E_Pair

class SE3EquivariantMessagePassingLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_vec: int,
        num_rbf: int = 16,
        rbf_gamma: float = 10.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_vec = d_vec
        self.num_rbf = num_rbf
        self.rbf_gamma = rbf_gamma

        self.register_buffer("rbf_centers", torch.linspace(0.0, 1.0, num_rbf))

        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d_model + num_rbf, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.msg_scalar = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.msg_vec_scale = nn.Sequential(
            nn.Linear(d_model, d_vec),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.h_norm = nn.LayerNorm(d_model)

    def _rbf(self, dist: torch.Tensor, dist_max: float):

        if dist.dim() == 3:
            dist = dist.squeeze(-1)
        centers = torch.linspace(0.0, float(dist_max), self.num_rbf, device=dist.device, dtype=dist.dtype)
        diff = dist.unsqueeze(-1) - centers.view(1, 1, -1)
        return torch.exp(-self.rbf_gamma * diff * diff)

    def forward(
        self,
        h: torch.Tensor,
        v: torch.Tensor,
        coords: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        dist_max: float,
    ):

        B, N, _ = h.shape
        E = int(edge_src.numel())
        if E == 0:
            return h, v

        src = edge_src
        dst = edge_dst

        coords_src = coords[:, src, :]
        coords_dst = coords[:, dst, :]
        rel = coords_dst - coords_src
        dist = torch.norm(rel, dim=-1, keepdim=True)
        rbf = self._rbf(dist, dist_max=dist_max)

        h_src = h[:, src, :]
        h_dst = h[:, dst, :]

        gate_in = torch.cat([h_src, h_dst, rbf], dim=-1)
        w = self.gate_mlp(gate_in)
        w = torch.sigmoid(w)

        msg_h = w * self.msg_scalar(h_src)

        msg_v_scale = w * self.msg_vec_scale(h_src)
        msg_v = msg_v_scale.unsqueeze(-1) * rel.unsqueeze(-2)

        h_new = h.clone()
        v_new = v.clone()

        for b in range(B):
            h_new[b].index_add_(0, dst, msg_h[b])
            v_new[b].index_add_(0, dst, msg_v[b])

        h_new = self.h_norm(h_new)
        return h_new, v_new

class SE3EquivariantCompositionEncoder(nn.Module):

    def __init__(
        self,
        component_graphs,
        d_model: int,
        d_vec: int = 8,
        num_rbf: int = 16,
        cutoff: float = 6.0,
        num_layers: int = 2,
        dropout: float = 0.1,
        seed: int = 0,
    ):
        super().__init__()
        self.component_graphs = component_graphs
        self.num_components = len(component_graphs)
        self.d_model = d_model
        self.d_vec = d_vec
        self.cutoff = cutoff

        self.max_atomic_num = 60
        self.atom_embedding = nn.Embedding(self.max_atomic_num + 1, d_model)

        self.mp_layers = nn.ModuleList(
            [
                SE3EquivariantMessagePassingLayer(
                    d_model=d_model,
                    d_vec=d_vec,
                    num_rbf=num_rbf,
                    rbf_gamma=10.0,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.v_init_scale = nn.Sequential(
            nn.Linear(d_model, d_vec),
            nn.GELU(),
        )

        self.comp_scalar_proj = nn.Sequential(
            nn.Linear(d_model + d_vec, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.comp_attn = nn.Linear(d_model, 1)

        self.coord_denoise_gamma = nn.Sequential(
            nn.Linear(d_model, d_vec),
            nn.GELU(),
            nn.Linear(d_vec, d_vec),
            nn.Tanh(),
        )

        self._precompute_identity()

        self._coords_clean_list = []
        self._atom_z_list = []
        self._edge_src_list = []
        self._edge_dst_list = []
        for c in range(self.num_components):
            g = self.component_graphs[c]
            self._coords_clean_list.append(torch.tensor(g["coords"], dtype=torch.float32))
            self._atom_z_list.append(torch.tensor(g["atom_z"], dtype=torch.long))
            self._edge_src_list.append(torch.tensor(g["edge_src"], dtype=torch.long))
            self._edge_dst_list.append(torch.tensor(g["edge_dst"], dtype=torch.long))

    def _precompute_identity(self):
        device = torch.device("cpu")
        comp_scalar_list = []
        comp_vec_list = []
        with torch.no_grad():
            for c in range(self.num_components):
                g = self.component_graphs[c]
                coords = torch.tensor(g["coords"], dtype=torch.float32, device=device)
                atom_z = torch.tensor(g["atom_z"], dtype=torch.long, device=device)
                edge_src = torch.tensor(g["edge_src"], dtype=torch.long, device=device)
                edge_dst = torch.tensor(g["edge_dst"], dtype=torch.long, device=device)

                h, U = self._encode_single_component(
                    mol_rotation=None,
                    coords=coords,
                    atom_z=atom_z,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                )
                comp_scalar_list.append(h.squeeze(0))
                comp_vec_list.append(U.squeeze(0))

        comp_scalar_id = torch.stack(comp_scalar_list, dim=0)
        comp_vec_id = torch.stack(comp_vec_list, dim=0)
        self.register_buffer("comp_scalar_id", comp_scalar_id)
        self.register_buffer("comp_vec_id", comp_vec_id)

    def _encode_single_component(
        self,
        mol_rotation: torch.Tensor,
        coords: torch.Tensor,
        atom_z: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ):

        if atom_z.numel() == 0:
            raise ValueError("component graph has no atoms")

        if mol_rotation is None:
            B = 1
            coords_b = coords.unsqueeze(0)
            h = self.atom_embedding(torch.clamp(atom_z, 0, self.max_atomic_num)).unsqueeze(0)
        else:
            B = mol_rotation.size(0)

            coords_b = torch.einsum("bij,nj->bni", mol_rotation, coords)
            h = self.atom_embedding(torch.clamp(atom_z, 0, self.max_atomic_num)).unsqueeze(0).expand(B, -1, -1)

        coords_centered = coords_b
        v_scale = self.v_init_scale(h)
        v = coords_centered.unsqueeze(-2) * v_scale.unsqueeze(-1)

        dist_max = float(self.cutoff)
        for layer in self.mp_layers:
            h, v = layer(h, v, coords_b, edge_src, edge_dst, dist_max=dist_max)

        h_mean = h.mean(dim=1)
        v_norm = torch.norm(v, dim=-1)
        v_mean_norm = v_norm.mean(dim=1)
        E_scalar = self.comp_scalar_proj(torch.cat([h_mean, v_mean_norm], dim=-1))

        attn_logits = self.comp_attn(h)
        attn = torch.softmax(attn_logits, dim=1)
        U_vec = (attn.unsqueeze(-1) * v).sum(dim=1)

        return E_scalar, U_vec

    def _encode_component_node_vectors(
        self,
        coords_b: torch.Tensor,
        atom_z: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> torch.Tensor:

        B, N, _ = coords_b.shape
        atom_z = torch.clamp(atom_z, 0, self.max_atomic_num)
        h = self.atom_embedding(atom_z).unsqueeze(0).expand(B, -1, -1)

        v_scale = self.v_init_scale(h)
        v = coords_b.unsqueeze(-2) * v_scale.unsqueeze(-1)

        dist_max = float(self.cutoff)
        for layer in self.mp_layers:
            h, v = layer(h, v, coords_b, edge_src, edge_dst, dist_max=dist_max)
        return v

    def denoise_loss(
        self,
        mol_fractions: torch.Tensor,
        formula_embedding: torch.Tensor,
        sigma: float = 0.3,
        weight_threshold: float = 1e-8,
        eps: float = 1e-8,
    ) -> torch.Tensor:

        device = mol_fractions.device
        B, C = mol_fractions.shape
        assert C == self.num_components

        gamma = self.coord_denoise_gamma(formula_embedding)

        total_loss = torch.zeros((), device=device, dtype=formula_embedding.dtype)
        active_components = 0

        for c in range(self.num_components):
            w = mol_fractions[:, c]
            idx = (w > weight_threshold).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue

            coords_clean = self._coords_clean_list[c].to(device)
            atom_z = self._atom_z_list[c].to(device)
            edge_src = self._edge_src_list[c].to(device)
            edge_dst = self._edge_dst_list[c].to(device)

            coords_clean_b = coords_clean.unsqueeze(0).expand(idx.numel(), -1, -1)
            noise = torch.randn_like(coords_clean_b) * float(sigma)
            coords_noisy = coords_clean_b + noise

            coords_noisy = coords_noisy - coords_noisy.mean(dim=1, keepdim=True)

            v = self._encode_component_node_vectors(
                coords_b=coords_noisy,
                atom_z=atom_z,
                edge_src=edge_src,
                edge_dst=edge_dst,
            )

            gamma_idx = gamma[idx]

            delta = torch.einsum("bd,bndj->bnj", gamma_idx, v)

            coords_denoise = coords_noisy + delta
            diff = coords_denoise - coords_clean_b

            diff_sq = diff.pow(2).sum(dim=(1, 2))
            loss_c = (w[idx] * diff_sq).sum() / (w[idx].sum() + eps)

            total_loss = total_loss + loss_c
            active_components += 1

        if active_components == 0:
            return torch.zeros((), device=device, dtype=formula_embedding.dtype)

        return total_loss / active_components

    def forward(self, mol_fractions: torch.Tensor, se3_rotation: torch.Tensor = None):

        B, C = mol_fractions.shape
        assert C == self.num_components

        if se3_rotation is None:

            E_mix = torch.matmul(mol_fractions, self.comp_scalar_id.to(mol_fractions.device))
            U_mix = torch.einsum("bc,cvj->bvj", mol_fractions, self.comp_vec_id.to(mol_fractions.device))
            return E_mix, U_mix

        E_mix = torch.zeros(B, self.d_model, device=mol_fractions.device, dtype=mol_fractions.dtype)
        U_mix = torch.zeros(B, self.d_vec, 3, device=mol_fractions.device, dtype=mol_fractions.dtype)

        for c in range(self.num_components):
            g = self.component_graphs[c]
            coords = torch.tensor(g["coords"], dtype=torch.float32, device=mol_fractions.device)
            atom_z = torch.tensor(g["atom_z"], dtype=torch.long, device=mol_fractions.device)
            edge_src = torch.tensor(g["edge_src"], dtype=torch.long, device=mol_fractions.device)
            edge_dst = torch.tensor(g["edge_dst"], dtype=torch.long, device=mol_fractions.device)

            E_comp, U_comp = self._encode_single_component(
                mol_rotation=se3_rotation,
                coords=coords,
                atom_z=atom_z,
                edge_src=edge_src,
                edge_dst=edge_dst,
            )

            w = mol_fractions[:, c].unsqueeze(-1)
            E_mix = E_mix + w * E_comp
            U_mix = U_mix + w.unsqueeze(-1) * U_comp

        return E_mix, U_mix

class RDFPretrainModel(nn.Module):

    def __init__(self, config, component_graphs=None):
        super().__init__()
        self.config = config
        
        self.component_encoder = ComponentFeatureEncoder(
            num_components=config['num_components'],
            hidden_dim=config['component_hidden_dim'],
            output_dim=config['d_model'],
            dropout=config['dropout']
        )

        self.use_se3_equiv = bool(config.get('use_se3_equivariance', False))
        if self.use_se3_equiv:
            if component_graphs is None:
                raise ValueError("component_graphs must not be None when use_se3_equivariance=True.")
            self.se3_composition_encoder = SE3EquivariantCompositionEncoder(
                component_graphs=component_graphs,
                d_model=config['d_model'],
                d_vec=int(config.get('se3_d_vec', 8)),
                num_rbf=int(config.get('se3_num_rbf', 16)),
                cutoff=float(config.get('se3_cutoff', 6.0)),
                num_layers=int(config.get('se3_num_layers', 2)),
                dropout=float(config['dropout']),
            )
            self.formula_se3_fusion = nn.Sequential(
                nn.Linear(config['d_model'] * 2, config['d_model']),
                nn.LayerNorm(config['d_model']),
                nn.GELU(),
                nn.Dropout(config['dropout']),
            )
        
        self.ion_pair_embedding = nn.Embedding(
            num_embeddings=config['num_ion_pairs'],
            embedding_dim=config['d_model']
        )
        
        self.position_encoder = PositionalEncoding(
            d_model=config['d_model'],
            max_len=config['max_seq_len']
        )
        
        self.input_fusion = nn.Sequential(
            nn.Linear(config['d_model'] * 3, config['d_model']),
            nn.LayerNorm(config['d_model']),
            nn.GELU(),
            nn.Dropout(config['dropout'])
        )
        
        self.rdf_transformer = RDFTransformerBlock(
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout']
        )

    def compute_formula_embedding(
        self,
        mol_fractions: torch.Tensor,
        se3_rotation: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        E_Formula_basic = self.component_encoder(mol_fractions)
        if self.use_se3_equiv:
            E_Formula_se3, U_mix = self.se3_composition_encoder(mol_fractions, se3_rotation=se3_rotation)
            E_Formula_Pre = self.formula_se3_fusion(torch.cat([E_Formula_basic, E_Formula_se3], dim=-1))
            return E_Formula_Pre, U_mix
        return E_Formula_basic, None

    def compute_denoise_loss(
        self,
        mol_fractions: torch.Tensor,
        sigma: float = 0.3,
    ) -> torch.Tensor:

        if not self.use_se3_equiv:
            return torch.zeros((), device=mol_fractions.device, dtype=mol_fractions.dtype)
        E_Formula_Pre, _ = self.compute_formula_embedding(mol_fractions, se3_rotation=None)
        return self.se3_composition_encoder.denoise_loss(
            mol_fractions=mol_fractions,
            formula_embedding=E_Formula_Pre,
            sigma=float(sigma),
        )
    
    def forward(
        self,
        mol_fractions,
        ion_pair_ids,
        r_positions,
        mask=None,
        se3_rotation: torch.Tensor = None,
        return_se3_vector: bool = False,
    ):
        batch_size, seq_len = ion_pair_ids.shape
        
        E_Formula_basic = self.component_encoder(mol_fractions)
        U_mix = None
        if self.use_se3_equiv:
            E_Formula_se3, U_mix = self.se3_composition_encoder(mol_fractions, se3_rotation=se3_rotation)
            E_Formula_Pre = self.formula_se3_fusion(torch.cat([E_Formula_basic, E_Formula_se3], dim=-1))
        else:
            E_Formula_Pre = E_Formula_basic
        E_Formula_Pre = E_Formula_Pre.unsqueeze(1).expand(-1, seq_len, -1)
        
        E_Pair_Type = self.ion_pair_embedding(ion_pair_ids)
        
        E_Space = torch.zeros_like(E_Pair_Type)
        E_Space = self.position_encoder(E_Space, r_positions)
        
        X_input = torch.cat([E_Pair_Type, E_Space, E_Formula_Pre], dim=-1)
        X_input = self.input_fusion(X_input)
        
        predictions, E_Pair = self.rdf_transformer(X_input, mask)

        if return_se3_vector:
            return predictions, E_Pair, U_mix
        return predictions, E_Pair

class RDFDatasetAugmented(Dataset):

    def __init__(self, csv_file, ion_pair_vocab, mol_columns=None, max_seq_len=1200, 
                 augment=False, noise_level=0.005):
        print(f"Loading dataset from {csv_file}...")
        self.data = pd.read_csv(csv_file)
        self.ion_pair_vocab = ion_pair_vocab
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.noise_level = noise_level
        
        if mol_columns is None:
            self.mol_columns = sorted([col for col in self.data.columns if col.startswith('mol_count_')])
        else:
            self.mol_columns = mol_columns
        
        self.groups = []
        for (folder_path, ion_pair), group_df in self.data.groupby(['folder_path', 'ion_pair']):
            if len(group_df) > 0:
                self.groups.append({
                    'folder_path': folder_path,
                    'ion_pair': ion_pair,
                    'data': group_df
                })
        
        print(f"  Total groups (system-ion_pair): {len(self.groups)}")
    
    def __len__(self):
        return len(self.groups)
    
    def __getitem__(self, idx):
        group = self.groups[idx]
        group_df = group['data']
        
        mol_counts = []
        for col in self.mol_columns:
            if col in group_df.columns:
                val = group_df[col].iloc[0]
                mol_counts.append(float(val) if pd.notna(val) else 0.0)
            else:
                mol_counts.append(0.0)
        mol_counts = np.array(mol_counts, dtype=np.float32)

        if self.augment and np.random.rand() < 0.3:
            noise = np.random.normal(0, self.noise_level, size=mol_counts.shape).astype(np.float32)
            mol_counts = (mol_counts + noise * mol_counts).astype(np.float32)
            mol_counts = np.clip(mol_counts, 0, None).astype(np.float32)
        
        total = mol_counts.sum()
        mol_fractions = (mol_counts / total if total > 0 else mol_counts).astype(np.float32)
        
        ion_pair = group['ion_pair']
        ion_pair_id = self.ion_pair_vocab.get(ion_pair, 0)
        
        r_values = group_df['r_nm'].values.astype(np.float32)
        g_r_values = group_df['g_r'].values.astype(np.float32)

        
        seq_len = min(len(r_values), self.max_seq_len)
        r_positions = (r_values[:seq_len] / 0.002).astype(np.int64)
        r_positions = np.clip(r_positions, 0, 4999)
        
        ion_pair_ids = np.full(self.max_seq_len, ion_pair_id, dtype=np.int64)
        r_positions_padded = np.zeros(self.max_seq_len, dtype=np.int64)
        g_r_padded = np.zeros(self.max_seq_len, dtype=np.float32)
        mask = np.ones(self.max_seq_len, dtype=bool)
        
        ion_pair_ids[:seq_len] = ion_pair_id
        r_positions_padded[:seq_len] = r_positions
        g_r_padded[:seq_len] = g_r_values[:seq_len]
        mask[:seq_len] = False
        
        return {
            'mol_fractions': torch.from_numpy(mol_fractions),
            'ion_pair_ids': torch.from_numpy(ion_pair_ids),
            'r_positions': torch.from_numpy(r_positions_padded),
            'g_r': torch.from_numpy(g_r_padded),
            'mask': torch.from_numpy(mask),
            'seq_len': seq_len
        }

def compute_loss(predictions, targets, mask):

    predictions = predictions.squeeze(-1)
    valid_mask = ~mask
    loss = F.mse_loss(predictions[valid_mask], targets[valid_mask])
    return loss

def build_ion_pair_vocab(csv_files):
    print("Building ion pair vocabulary...")
    ion_pairs = set()
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        ion_pairs.update(df['ion_pair'].unique())
    ion_pair_vocab = {ip: idx for idx, ip in enumerate(sorted(ion_pairs))}
    print(f"  Total ion pair types: {len(ion_pair_vocab)}")
    return ion_pair_vocab

def get_all_mol_columns(csv_files):
    print("Collecting molecular component columns...")
    mol_columns_set = set()
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, nrows=1)
        cols = [col for col in df.columns if col.startswith('mol_count_')]
        mol_columns_set.update(cols)
    mol_columns = sorted(list(mol_columns_set))
    print(f"  Total molecular components: {len(mol_columns)}")
    return mol_columns

def train_epoch(model, dataloader, optimizer, device, epoch, config):
    model.train()
    total_loss = 0

    use_se3 = bool(config.get('use_se3_equivariance', False))
    se3_every = int(config.get('se3_consistency_every', 2))
    se3_pred_weight = float(config.get('se3_pred_consistency_weight', 0.1))
    se3_vec_weight = float(config.get('se3_vec_equivariance_weight', 0.1))

    use_denoise = bool(config.get('use_se3_denoise', True))
    se3_denoise_every = int(config.get('se3_denoise_every', se3_every))
    se3_denoise_sigma = float(config.get('se3_denoise_sigma', 0.3))
    se3_denoise_gamma = float(config.get('se3_denoise_gamma', 0.1))

    for batch_idx, batch in enumerate(dataloader):
        mol_fractions = batch['mol_fractions'].to(device)
        ion_pair_ids = batch['ion_pair_ids'].to(device)
        r_positions = batch['r_positions'].to(device)
        g_r = batch['g_r'].to(device)
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()

        if use_se3:
            predictions, E_Pair, U_mix = model(
                mol_fractions,
                ion_pair_ids,
                r_positions,
                mask,
                return_se3_vector=True,
            )
        else:
            predictions, E_Pair = model(mol_fractions, ion_pair_ids, r_positions, mask)
            U_mix = None

        loss_rdf = compute_loss(predictions, g_r, mask)
        loss = loss_rdf

        if use_se3 and (batch_idx % se3_every == 0) and (U_mix is not None):
            B = mol_fractions.size(0)
            R = _random_rotation_matrix(B, device=device)

            predictions_rot, _, U_mix_rot = model(
                mol_fractions,
                ion_pair_ids,
                r_positions,
                mask,
                se3_rotation=R,
                return_se3_vector=True,
            )

            pred1 = predictions.squeeze(-1)
            pred2 = predictions_rot.squeeze(-1)
            valid = ~mask
            loss_pred_cons = F.mse_loss(pred1[valid], pred2[valid])

            U_expected = _rotate_vecs(R, U_mix)
            loss_vec_equiv = F.mse_loss(U_mix_rot, U_expected)

            loss = loss_rdf + se3_pred_weight * loss_pred_cons + se3_vec_weight * loss_vec_equiv

        if use_se3 and use_denoise and (batch_idx % se3_denoise_every == 0):

            loss_denoise = model.compute_denoise_loss(
                mol_fractions=mol_fractions,
                sigma=se3_denoise_sigma,
            )
            loss = loss + se3_denoise_gamma * loss_denoise
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_mse = 0
    total_mae = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            mol_fractions = batch['mol_fractions'].to(device)
            ion_pair_ids = batch['ion_pair_ids'].to(device)
            r_positions = batch['r_positions'].to(device)
            g_r = batch['g_r'].to(device)
            mask = batch['mask'].to(device)
            
            predictions, E_Pair = model(mol_fractions, ion_pair_ids, r_positions, mask)
            
            loss = compute_loss(predictions, g_r, mask)
            total_loss += loss.item()
            
            predictions = predictions.squeeze(-1)
            valid_mask = ~mask
            
            mse = F.mse_loss(predictions[valid_mask], g_r[valid_mask], reduction='sum')
            mae = F.l1_loss(predictions[valid_mask], g_r[valid_mask], reduction='sum')
            
            total_mse += mse.item()
            total_mae += mae.item()
            total_samples += valid_mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    rmse = np.sqrt(total_mse / total_samples)
    mae = total_mae / total_samples
    
    return avg_loss, rmse, mae

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
_PRETRAIN_DIR = _REPO_ROOT / "src" / "pretrain"
_DEFAULT_CONFIG_PATH = _PRETRAIN_DIR / "train_config.yaml"
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "pretrain"
_DEFAULT_SAVE_DIR = _REPO_ROOT / "data" / "checkpoints"

# Fallback values, used only when a key is absent from the YAML config.
_DEFAULT_CONFIG = {
    'train_csv': 'pretrain_rdf_train.csv',
    'val_csv': 'pretrain_rdf_val.csv',
    'test_csv': 'pretrain_rdf_test.csv',

    'num_components': 12,
    'num_ion_pairs': 20,
    'component_hidden_dim': 128,
    'd_model': 256,
    'nhead': 8,
    'num_layers': 5,
    'dim_feedforward': 768,
    'dropout': 0.15,
    'max_seq_len': 1200,

    'batch_size': 24,
    'learning_rate': 7e-5,
    'weight_decay': 5e-5,
    'num_epochs': 80,
    'warmup_epochs': 5,

    'augment_train': True,
    'noise_level': 0.005,

    'use_se3_equivariance': True,
    'se3_consistency_every': 2,
    'se3_pred_consistency_weight': 0.1,
    'se3_vec_equivariance_weight': 0.1,
    'se3_d_vec': 8,
    'se3_num_rbf': 16,
    'se3_cutoff': 6.0,
    'se3_num_layers': 2,
    'rdkit_seed': 0,

    'use_se3_denoise': True,
    'se3_denoise_every': 2,
    'se3_denoise_sigma': 0.3,
    'se3_denoise_gamma': 0.1,

    'early_stopping_patience': 12,
    'min_delta': 0.0005,

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': str(_DEFAULT_SAVE_DIR),
    'save_interval': 10,
    'seed': 42,
}

_CONFIG_SECTIONS = ('data', 'model', 'training', 'save')


def _flatten_config(raw: dict) -> dict:
    """Flatten the nested YAML sections into the flat config dict used below."""
    if not isinstance(raw, dict):
        raise ValueError("train config must be a mapping")
    flat: dict = {}
    for section in _CONFIG_SECTIONS:
        block = raw.get(section)
        if isinstance(block, dict):
            flat.update(block)
    for key, value in raw.items():
        if key not in _CONFIG_SECTIONS and not isinstance(value, dict):
            flat[key] = value
    return flat


def load_train_config(config_path: str | os.PathLike | None = None) -> dict:
    """Load pretraining config from YAML, falling back to the built-in defaults.

    Precedence: built-in defaults < YAML file < CLI overrides (applied in ``main``).
    """
    config = dict(_DEFAULT_CONFIG)
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if Path(path).is_file():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        config.update(_flatten_config(raw))
    elif config_path:
        raise FileNotFoundError(f"Pretraining config not found: {config_path}")
    return config


def _resolve_pretrain_paths(config: dict, data_dir: str | os.PathLike | None,
                           save_dir: str | os.PathLike | None) -> dict:
    data_root = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    for key in ('train_csv', 'val_csv', 'test_csv'):
        p = Path(config[key])
        config[key] = str(p if p.is_absolute() else data_root / p)

    save_root = Path(save_dir) if save_dir else Path(config['save_dir'])
    if not save_root.is_absolute():
        # Relative save paths are resolved against the repository root,
        # not the caller's current working directory.
        save_root = (_REPO_ROOT / save_root).resolve()
    config['save_dir'] = str(save_root)
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage I: RDF pretraining. Defaults come from src/pretrain/train_config.yaml."
    )
    parser.add_argument("--config", type=str, default=None,
                        help=f"YAML config (default: {_DEFAULT_CONFIG_PATH})")
    parser.add_argument("--data-dir", type=str, default=None,
                        help=f"Directory containing the pretrain CSVs (default: {_DEFAULT_DATA_DIR})")
    parser.add_argument("--save-dir", type=str, default=None,
                        help=f"Checkpoint output directory (default: {_DEFAULT_SAVE_DIR})")
    parser.add_argument("--device", type=str, default=None, choices=("cuda", "cpu"),
                        help="Override the device in the config")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.num_epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override training.batch_size")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    print("="*70)
    print("Stage I: structural pretraining encoder training (balanced)")
    print("="*70)

    config = load_train_config(args.config)
    config = _resolve_pretrain_paths(config, args.data_dir, args.save_dir)
    if args.device is not None:
        config['device'] = args.device
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    random.seed(int(config['seed']))
    np.random.seed(int(config['seed']))
    torch.manual_seed(int(config['seed']))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config['seed']))

    missing = [config[k] for k in ('train_csv', 'val_csv', 'test_csv') if not os.path.isfile(config[k])]
    if missing:
        raise FileNotFoundError(
            "Pretraining CSV not found: " + ", ".join(missing) +
            f"\nExpected them under {_DEFAULT_DATA_DIR} (or pass --data-dir)."
        )

    os.makedirs(config['save_dir'], exist_ok=True)
    print(f"\nConfig: {args.config or _DEFAULT_CONFIG_PATH}")
    print(f"Data dir: {_DEFAULT_DATA_DIR if not args.data_dir else args.data_dir}")
    print(f"Save dir: {config['save_dir']}")
    
    print(f"\nDevice: {config['device']}")
    print("Balanced strategy:")
    print("  - Moderate model capacity (layers: 5, ff: 768)")
    print("  - Moderate regularization (dropout: 0.15, weight_decay: 5e-5)")
    print("  - Balanced learning rate (7e-5)")
    print("  - Light data augmentation (noise: 0.005, composition only)")
    print("  - Early stopping (patience: 12)")
    print("  - GELU activation")
    
    csv_files = [config['train_csv'], config['val_csv'], config['test_csv']]
    ion_pair_vocab = build_ion_pair_vocab(csv_files)
    config['num_ion_pairs'] = len(ion_pair_vocab)
    
    mol_columns = get_all_mol_columns(csv_files)
    config['num_components'] = len(mol_columns)
    
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(config['save_dir'], 'ion_pair_vocab.json'), 'w') as f:
        json.dump(ion_pair_vocab, f, indent=2, ensure_ascii=False)
    with open(os.path.join(config['save_dir'], 'mol_columns.json'), 'w') as f:
        json.dump(mol_columns, f, indent=2, ensure_ascii=False)
    
    print("\nCreating dataset...")
    train_dataset = RDFDatasetAugmented(
        config['train_csv'], ion_pair_vocab, mol_columns, config['max_seq_len'],
        augment=config['augment_train'], noise_level=config['noise_level']
    )
    val_dataset = RDFDatasetAugmented(
        config['val_csv'], ion_pair_vocab, mol_columns, config['max_seq_len'],
        augment=False
    )
    test_dataset = RDFDatasetAugmented(
        config['test_csv'], ion_pair_vocab, mol_columns, config['max_seq_len'],
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"  Train: {len(train_dataset)} groups")
    print(f"  Validation: {len(val_dataset)} groups")
    print(f"  Test: {len(test_dataset)} groups")
    
    print("\nCreating model...")
    component_graphs = None
    if config.get('use_se3_equivariance', False):
        print("  Generating component 3D coords and intra-molecular graphs (RDKit)...")
        component_graphs = build_component_graphs_from_rdkit(
            mol_columns,
            cutoff=float(config.get('se3_cutoff', 6.0)),
            seed=int(config.get('rdkit_seed', 0)),
        )
    model = RDFPretrainModel(config, component_graphs=component_graphs).to(config['device'])
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )

    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return (epoch + 1) / config['warmup_epochs']
        else:
            progress = (epoch - config['warmup_epochs']) / (config['num_epochs'] - config['warmup_epochs'])
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print("\nStarting training...")
    best_val_rmse = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_rmses = []
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print('='*70)
        
        train_loss = train_epoch(model, train_loader, optimizer, config['device'], epoch, config)
        print(f"Train loss: {train_loss:.4f}")

        val_loss, val_rmse, val_mae = evaluate(model, val_loader, config['device'])
        print(f"Val loss: {val_loss:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.2e}")
        scheduler.step()
        
        train_losses.append(train_loss)
        val_rmses.append(val_rmse)
        
        if val_rmse < best_val_rmse - config['min_delta']:
            best_val_rmse = val_rmse
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse,
                'config': config
            }, os.path.join(config['save_dir'], 'best_model.pt'))
            print(f"✓ Saved best model (RMSE: {val_rmse:.4f})")
        else:
            patience_counter += 1
            print(f"Val RMSE did not improve ({patience_counter}/{config['early_stopping_patience']})")

        if patience_counter >= config['early_stopping_patience']:
            print(f"\nEarly stopping triggered: validation RMSE did not improve for {config['early_stopping_patience']} epochs")
            break
        
        if epoch % config['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse,
                'config': config
            }, os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch}.pt'))
    
    print("\n" + "="*70)
    print("Testing best model")
    print("="*70)

    checkpoint = torch.load(os.path.join(config['save_dir'], 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_rmse, test_mae = evaluate(model, test_loader, config['device'])
    print(f"Test loss: {test_loss:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

    print("\n✓ Training complete!")
    print(f"Best val RMSE: {best_val_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Overfitting ratio: {test_rmse/best_val_rmse:.2f}x")

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Best val RMSE: {best_val_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Overfitting ratio: {test_rmse/best_val_rmse:.2f}x")

    np.save(os.path.join(config['save_dir'], 'train_losses.npy'), train_losses)
    np.save(os.path.join(config['save_dir'], 'val_rmses.npy'), val_rmses)

if __name__ == '__main__':
    main()
