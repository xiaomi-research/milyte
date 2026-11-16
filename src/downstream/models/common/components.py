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

COMPONENT_PROPERTIES = {
    'EC': {'Dielectric_Constant': 89.6, "Solvation_Ratio": 1.21, "Solvation_Power": 0.77, 'Solvation_G': -2.0755},
    'EMC': {'Dielectric_Constant': 2.9, "Solvation_Ratio": 0.99, "Solvation_Power": 0.83, 'Solvation_G': -1.889},
    'DMC': {'Dielectric_Constant': 3.2, "Solvation_Ratio": 1.05, "Solvation_Power": 0.85, 'Solvation_G': -1.8253},
    'DEC': {'Dielectric_Constant': 2.8, "Solvation_Ratio": 0.94, "Solvation_Power": 0.73, 'Solvation_G': -1.9496},
    'PC': {'Dielectric_Constant': 69, "Solvation_Ratio": 1.26, "Solvation_Power": 0.99, 'Solvation_G': -2.14012},
    'MA': {'Dielectric_Constant': 7.3, "Solvation_Ratio": 1.22, "Solvation_Power": 1.396, 'Solvation_G': -1.94592},
    'EA': {'Dielectric_Constant': 6.4, "Solvation_Ratio": 1.18, "Solvation_Power": 1.46, 'Solvation_G': -2.00827},
    'EP': {'Dielectric_Constant': 5, "Solvation_Ratio": 1.05, "Solvation_Power": 0.967, 'Solvation_G': -2.02505},
    'PP': {'Dielectric_Constant': 4.7, "Solvation_Ratio": 1, "Solvation_Power": 1, 'Solvation_G': -2.04269},
    'VN': {'Dielectric_Constant': 24.7, "Solvation_Ratio": 1.23, "Solvation_Power": 2.53, 'Solvation_G': -1.97309},
    'LiPF6': {'Dielectric_Constant': 0.0, "Solvation_Ratio": 0.0, "Solvation_Power": 0.0, 'Solvation_G': 0.0},
    'LiFSI': {'Dielectric_Constant': 0.0, "Solvation_Ratio": 0.0, "Solvation_Power": 0.0, 'Solvation_G': 0.0},
}

PROPERTY_NAMES = ['Dielectric_Constant', 'Solvation_Ratio', 'Solvation_Power', 'Solvation_G']

def extract_physicochemical_features(mol_columns, mol_fractions):

    batch_size = mol_fractions.shape[0]
    num_properties = len(PROPERTY_NAMES)

    component_names = [col.replace('mol_count_', '').upper() for col in mol_columns]

    prop_matrix = []
    for comp_name in component_names:
        if comp_name in COMPONENT_PROPERTIES:
            props = COMPONENT_PROPERTIES[comp_name]
            prop_vector = [props.get(p, 0.0) for p in PROPERTY_NAMES]
        else:
            prop_vector = [0.0] * num_properties
        prop_matrix.append(prop_vector)
    
    prop_matrix = torch.tensor(prop_matrix, dtype=torch.float32, device=mol_fractions.device)

    weighted_props = torch.matmul(mol_fractions, prop_matrix)
    
    return weighted_props
