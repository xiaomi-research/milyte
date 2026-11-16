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
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .components import COMPONENT_PROPERTIES, PROPERTY_NAMES, extract_physicochemical_features

class ArrheniusTemperatureEmbedding(nn.Module):

    R_J_PER_MOL_K = 8.314

    def __init__(self, num_ea: int = 50, ea_min_kj_per_mol: float = 10.0, ea_max_kj_per_mol: float = 100.0):
        super().__init__()
        self.num_ea = num_ea
        ea_j = torch.linspace(ea_min_kj_per_mol, ea_max_kj_per_mol, num_ea) * 1000.0
        self.register_buffer("Ea_grid", ea_j)
        self.out_norm = nn.LayerNorm(num_ea)

    def forward(self, T_k: torch.Tensor) -> torch.Tensor:

        if not torch.is_tensor(T_k):
            T_k = torch.tensor(T_k, dtype=torch.float32, device=self.Ea_grid.device)
        T = T_k.reshape(-1, 1).to(dtype=self.Ea_grid.dtype, device=self.Ea_grid.device)
        T = torch.clamp(T, min=1.0)
        R = self.R_J_PER_MOL_K

        arrhenius_features = torch.exp(-self.Ea_grid.unsqueeze(0) / (R * T))
        return self.out_norm(arrhenius_features)

class EnhancedCompositionFeatureExtractor(nn.Module):

    def __init__(self, num_components, mol_columns, output_dim=128):
        super().__init__()
        self.mol_columns = mol_columns
        self.num_properties = len(PROPERTY_NAMES)

        self.fraction_encoder = nn.Sequential(
            nn.Linear(num_components, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )

        self.stats_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32)
        )

        self.property_encoder = nn.Sequential(
            nn.Linear(self.num_properties, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 64)
        )

        num_interaction_terms = (self.num_properties * (self.num_properties - 1)) // 2
        self.property_interaction_encoder = nn.Sequential(
            nn.Linear(num_interaction_terms, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 32)
        )

        self.fraction_property_interaction = nn.Sequential(
            nn.Linear(num_components + self.num_properties, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )

        total_features = 64 + 32 + 64 + 32 + 32
        self.attention = nn.Sequential(
            nn.Linear(total_features, total_features // 2),
            nn.Tanh(),
            nn.Linear(total_features // 2, total_features),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Linear(total_features, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
    
    def compute_property_interactions(self, physicochemical_props):

        batch_size = physicochemical_props.shape[0]
        interactions = []

        for i in range(self.num_properties):
            for j in range(i + 1, self.num_properties):
                interaction = physicochemical_props[:, i] * physicochemical_props[:, j]
                interactions.append(interaction.unsqueeze(1))
        
        if interactions:
            return torch.cat(interactions, dim=1)
        else:
            return torch.zeros(batch_size, 1, device=physicochemical_props.device)
    
    def forward(self, mol_fractions, mol_counts):

        fraction_feat = self.fraction_encoder(mol_fractions)

        total_mols = mol_counts.sum(dim=1, keepdim=True)
        salt_counts = mol_counts[:, -2:].sum(dim=1, keepdim=True)
        solvent_counts = mol_counts[:, :-2].sum(dim=1, keepdim=True)
        salt_ratio = salt_counts / (total_mols + 1e-8)
        solvent_ratio = solvent_counts / (total_mols + 1e-8)
        num_components = (mol_counts > 0).sum(dim=1, keepdim=True).float()
        num_salts = (mol_counts[:, -2:] > 0).sum(dim=1, keepdim=True).float()
        
        stats = torch.cat([
            torch.log1p(total_mols),
            salt_ratio,
            solvent_ratio,
            num_components,
            num_salts
        ], dim=1)
        
        stats_feat = self.stats_encoder(stats)

        physicochemical_props = extract_physicochemical_features(self.mol_columns, mol_fractions)
        property_feat = self.property_encoder(physicochemical_props)

        property_interactions = self.compute_property_interactions(physicochemical_props)
        property_interaction_feat = self.property_interaction_encoder(property_interactions)

        fraction_property = torch.cat([mol_fractions, physicochemical_props], dim=1)
        fraction_property_feat = self.fraction_property_interaction(fraction_property)

        all_features = torch.cat([
            fraction_feat,
            stats_feat,
            property_feat,
            property_interaction_feat,
            fraction_property_feat
        ], dim=1)

        attention_weights = self.attention(all_features)
        attended_features = all_features * attention_weights

        output = self.fusion(attended_features)
        
        return output

class FeatureModulationFusion(nn.Module):

    def __init__(self, structure_dim, composition_dim):
        super().__init__()

        self.modulation_net = nn.Sequential(
            nn.Linear(composition_dim, composition_dim),
            nn.LayerNorm(composition_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.scale_net = nn.Linear(composition_dim, structure_dim)
        self.bias_net = nn.Linear(composition_dim, structure_dim)

        self.gate_net = nn.Sequential(
            nn.Linear(composition_dim + structure_dim, structure_dim),
            nn.Sigmoid()
        )
    
    def forward(self, E_structure, E_composition):

        composition_processed = self.modulation_net(E_composition)

        scale = self.scale_net(composition_processed)
        bias = self.bias_net(composition_processed)

        E_modulated = E_structure * (1 + scale) + bias

        gate_input = torch.cat([E_composition, E_modulated], dim=1)
        gate = self.gate_net(gate_input)
        
        E_output = gate * E_modulated + (1 - gate) * E_structure
        
        return E_output

class RegressionHead(nn.Module):

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)
