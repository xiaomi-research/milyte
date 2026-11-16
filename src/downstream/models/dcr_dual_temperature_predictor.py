#!/usr/bin/env python3
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
"""
Dual-temperature DCR prediction: same backbone path as the conductivity/viscosity models

  - DCR @ -20 C (mOhm)
  - DCR @ 25 C (mOhm)


"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from common import EnhancedCompositionFeatureExtractor, FeatureModulationFusion
from rdf_backbone import RDFPretrainModel, build_component_graphs_from_rdkit

class RegressionHead2D(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        out_dim: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class DCRDualTemperaturePredictor(nn.Module):

    def __init__(
        self,
        pretrained_model: RDFPretrainModel,
        num_components: int,
        mol_columns: list[str],
        *,
        freeze_encoder: bool = True,
        composition_dim: int = 128,
        regression_hidden_dims: list[int] | None = None,
        num_prop_features: int = 3,
        use_prop_features: bool = True,
    ):
        super().__init__()
        self.mol_columns = mol_columns
        self.num_prop_features = num_prop_features
        self.use_prop_features = use_prop_features

        self.pretrained_model = pretrained_model
        if freeze_encoder:
            for p in self.pretrained_model.parameters():
                p.requires_grad = False

        self.composition_extractor = EnhancedCompositionFeatureExtractor(
            num_components=num_components,
            mol_columns=mol_columns,
            output_dim=composition_dim,
        )
        structure_dim = int(pretrained_model.config["d_model"])
        self.feature_modulation = FeatureModulationFusion(
            structure_dim=structure_dim,
            composition_dim=composition_dim,
        )
        self.prop_embed_dim = 32
        if use_prop_features:
            self.prop_branch = nn.Sequential(
                nn.LayerNorm(num_prop_features),
                nn.Linear(num_prop_features, self.prop_embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            in_dim = structure_dim + self.prop_embed_dim
        else:
            self.prop_branch = None
            in_dim = structure_dim
        self.regression_head = RegressionHead2D(
            input_dim=in_dim,
            hidden_dims=regression_hidden_dims or [256, 128, 64],
            out_dim=2,
            dropout=0.2,
        )

    def forward(
        self,
        mol_fractions: torch.Tensor,
        mol_counts: torch.Tensor,
        ion_pair_ids: torch.Tensor,
        r_positions: torch.Tensor,
        mask: torch.Tensor | None,
        prop_vec: torch.Tensor,
    ) -> torch.Tensor:

        encoder_frozen = all(not p.requires_grad for p in self.pretrained_model.parameters())
        grad_through_pretrained = (not encoder_frozen) or mol_fractions.requires_grad
        with torch.set_grad_enabled(grad_through_pretrained):
            _pred, E_structure = self.pretrained_model(
                mol_fractions, ion_pair_ids, r_positions, mask
            )
        E_composition = self.composition_extractor(mol_fractions, mol_counts)
        E_modulated = self.feature_modulation(E_structure, E_composition)
        if self.prop_branch is not None:
            p = self.prop_branch(prop_vec)
            h = torch.cat([E_modulated, p], dim=-1)
        else:
            h = E_modulated
        return self.regression_head(h)

class DCRDualDataset(Dataset):

    def __init__(
        self,
        data: list[dict[str, Any]],
        mol_columns: list[str],
        ion_pair_vocab: dict,
        max_seq_len: int = 1200,
    ):
        self.data = data
        self.mol_columns = mol_columns
        self.ion_pair_vocab = ion_pair_vocab
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.data[idx]
        atom_counts = sample["atom_counts"]
        mol_counts = np.zeros(len(self.mol_columns), dtype=np.float32)
        for i, col in enumerate(self.mol_columns):
            mol_name = col.replace("mol_count_", "").lower()
            if mol_name in atom_counts:
                mol_counts[i] = float(atom_counts[mol_name])
        total = mol_counts.sum()
        mol_fractions = (mol_counts / total if total > 0 else mol_counts).astype(np.float32)

        ion_pair_id = 0
        seq_len = 100
        ion_pair_ids = np.full(self.max_seq_len, ion_pair_id, dtype=np.int64)
        r_positions_padded = np.zeros(self.max_seq_len, dtype=np.int64)
        r_positions_padded[:seq_len] = np.arange(seq_len, dtype=np.int64)
        mask = np.ones(self.max_seq_len, dtype=bool)
        mask[:seq_len] = False

        y = np.asarray(sample["y_dcr_mohm"], dtype=np.float32).reshape(2)
        pv = np.asarray(sample["prop_vec"], dtype=np.float32).reshape(3)

        return {
            "mol_fractions": torch.from_numpy(mol_fractions),
            "mol_counts": torch.from_numpy(mol_counts),
            "ion_pair_ids": torch.from_numpy(ion_pair_ids),
            "r_positions": torch.from_numpy(r_positions_padded),
            "mask": torch.from_numpy(mask),
            "y_dcr": torch.from_numpy(y),
            "prop_vec": torch.from_numpy(pv),
            "ratio_str": sample.get("ratio_str", ""),
        }

def _row_atom_counts(row: pd.Series, mol_columns: list[str]) -> dict[str, float]:
    atom_counts: dict[str, float] = {}
    for col in mol_columns:
        comp = col.replace("mol_count_", "")
        if comp in row.index:
            v = pd.to_numeric(row[comp], errors="coerce")
            atom_counts[comp.lower()] = float(v) if pd.notna(v) else 0.0
        elif col in row.index:
            v = pd.to_numeric(row[col], errors="coerce")
            atom_counts[comp.replace("mol_count_", "").lower()] = float(v) if pd.notna(v) else 0.0
        else:
            atom_counts[comp.replace("mol_count_", "").lower()] = 0.0
    return atom_counts

def build_dcr_samples_from_component_csv(
    csv_path: str | os.PathLike,
    mol_columns: list[str],
    *,
    cold_col: str = "DCR_mOhm_minus20C",
    warm_col: str = "DCR_mOhm_25C",
    cond_col: str = "conductivity_mS_cm_mean5fold",
    visc_col: str = "viscosity_cP_mean5fold",
    g_col: str = "G_kJ_per_mol_mean5fold",
) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    for c in (cold_col, warm_col, cond_col, visc_col, g_col):
        if c not in df.columns:
            raise ValueError(f"CSV missing column: {c} (current columns: {list(df.columns)})")
    samples: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        ac = _row_atom_counts(row, mol_columns)
        ratio_str = "_".join(f"{k}:{v:.6f}" for k, v in sorted(ac.items()))
        cond_f = float(pd.to_numeric(row[cond_col], errors="coerce"))
        samples.append(
            {
                "atom_counts": ac,
                "ratio_str": ratio_str,

                "conductivity": cond_f,
                "y_dcr_mohm": [
                    float(pd.to_numeric(row[cold_col], errors="coerce")),
                    float(pd.to_numeric(row[warm_col], errors="coerce")),
                ],
                "prop_vec": [
                    cond_f,
                    float(pd.to_numeric(row[visc_col], errors="coerce")),
                    float(pd.to_numeric(row[g_col], errors="coerce")),
                ],
            }
        )
    return samples

def apply_fold_normalization(
    samples: list[dict[str, Any]],
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    props = np.array([[samples[i]["prop_vec"][j] for j in range(3)] for i in indices], dtype=np.float64)
    ys = np.array([[samples[i]["y_dcr_mohm"][j] for j in range(2)] for i in indices], dtype=np.float64)
    p_mean = props.mean(axis=0)
    p_std = props.std(axis=0) + 1e-6
    y_mean = ys.mean(axis=0)
    y_std = ys.std(axis=0) + 1e-6
    return p_mean, p_std, y_mean, y_std

def normalize_sample(
    s: dict[str, Any],
    p_mean: np.ndarray,
    p_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> dict[str, Any]:
    out = dict(s)
    pv = (np.array(s["prop_vec"], dtype=np.float64) - p_mean) / p_std
    yv = (np.array(s["y_dcr_mohm"], dtype=np.float64) - y_mean) / y_std
    out["prop_vec"] = pv.astype(np.float32).tolist()
    out["y_dcr_mohm"] = yv.astype(np.float32).tolist()
    return out

def _backbone_config_from_checkpoint(
    pretrained_model_path: str,
    mol_columns: list[str],
    device: torch.device,
) -> tuple[dict[str, Any], Any | None]:
    checkpoint = torch.load(pretrained_model_path, map_location=device, weights_only=False)
    if "config" in checkpoint:
        pretrained_config = checkpoint["config"]
    else:
        pretrained_config = {
            "d_model": 256,
            "nhead": 8,
            "num_layers": 5,
            "dim_feedforward": 768,
            "dropout": 0.15,
            "max_seq_len": 1200,
            "num_components": len(mol_columns),
            "num_ion_pairs": 20,
            "component_hidden_dim": 128,
        }
    component_graphs = None
    if pretrained_config.get("use_se3_equivariance", False):
        component_graphs = build_component_graphs_from_rdkit(
            mol_columns,
            cutoff=float(pretrained_config.get("se3_cutoff", 6.0)),
            seed=int(pretrained_config.get("rdkit_seed", 0)),
        )
    return pretrained_config, component_graphs

def load_pretrained_backbone_for_dcr(
    pretrained_model_path: str,
    mol_columns: list[str],
    device: torch.device,
) -> RDFPretrainModel:
    pretrained_config, component_graphs = _backbone_config_from_checkpoint(
        pretrained_model_path, mol_columns, device
    )
    m = RDFPretrainModel(pretrained_config, component_graphs=component_graphs).to(device)
    checkpoint = torch.load(pretrained_model_path, map_location=device, weights_only=False)
    m.load_state_dict(checkpoint["model_state_dict"])
    return m

def create_random_backbone_for_dcr(
    pretrained_model_path: str,
    mol_columns: list[str],
    device: torch.device,
) -> RDFPretrainModel:

    pretrained_config, component_graphs = _backbone_config_from_checkpoint(
        pretrained_model_path, mol_columns, device
    )
    return RDFPretrainModel(pretrained_config, component_graphs=component_graphs).to(device)
