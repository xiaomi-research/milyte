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

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class ConductivityDataset(Dataset):

    def __init__(self, data, mol_columns, ion_pair_vocab, max_seq_len=1200):
        self.data = data
        self.mol_columns = mol_columns
        self.ion_pair_vocab = ion_pair_vocab
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]

        atom_counts = sample['atom_counts']
        mol_counts = np.zeros(len(self.mol_columns), dtype=np.float32)
        
        for i, col in enumerate(self.mol_columns):
            mol_name = col.replace('mol_count_', '').lower()
            if mol_name in atom_counts:
                mol_counts[i] = float(atom_counts[mol_name])

        total = mol_counts.sum()
        mol_fractions = (mol_counts / total if total > 0 else mol_counts).astype(np.float32)

        ion_pair_id = 0

        seq_len = 100
        ion_pair_ids = np.full(self.max_seq_len, ion_pair_id, dtype=np.int64)
        r_positions = np.arange(seq_len, dtype=np.int64)
        r_positions_padded = np.zeros(self.max_seq_len, dtype=np.int64)
        r_positions_padded[:seq_len] = r_positions
        
        mask = np.ones(self.max_seq_len, dtype=bool)
        mask[:seq_len] = False

        conductivity = np.array([sample['conductivity']], dtype=np.float32)
        
        return {
            'mol_fractions': torch.from_numpy(mol_fractions),
            'mol_counts': torch.from_numpy(mol_counts),
            'ion_pair_ids': torch.from_numpy(ion_pair_ids),
            'r_positions': torch.from_numpy(r_positions_padded),
            'mask': torch.from_numpy(mask),
            'conductivity': torch.from_numpy(conductivity),
            'ratio_str': sample['ratio_str']
        }

def split_data_by_conductivity(data, target_conductivities, val_count=20,
                               samples_per_target=3, tolerance=0.5, random_state=42):

    print("\nUsing custom conductivity-based split strategy")
    print(f"  Target conductivity values: {sorted(target_conductivities)}")
    print(f"  Samples per target: {samples_per_target} closest")
    print(f"  Initial tolerance: ±{tolerance} mS/cm")

    test_data = []
    test_indices = set()
    
    for target_cond in target_conductivities:

        distances = []
        for idx, sample in enumerate(data):
            if idx not in test_indices:
                dist = abs(sample['conductivity'] - target_cond)
                if dist <= tolerance:
                    distances.append((dist, idx, sample))

        distances.sort(key=lambda x: x[0])
        selected = distances[:samples_per_target]
        
        for dist, idx, sample in selected:
            test_data.append(sample)
            test_indices.add(idx)
        
        if len(selected) > 0:
            print(f"  Target {target_cond:.2f}: selected {len(selected)} sample(s)")
        else:
            print(f"  Target {target_cond:.2f}: no sample within tolerance")

    remaining_data = [sample for idx, sample in enumerate(data) if idx not in test_indices]
    
    print(f"\n  Test set size: {len(test_data)}")
    if len(test_data) > 0:
        test_conds = [s['conductivity'] for s in test_data]
        print(f"  Test set conductivity range: [{min(test_conds):.2f}, {max(test_conds):.2f}]")
    print(f"  Remaining data: {len(remaining_data)} samples")

    np.random.seed(random_state)
    if len(remaining_data) >= val_count:
        val_indices = np.random.choice(len(remaining_data), size=val_count, replace=False)
        val_data = [remaining_data[i] for i in val_indices]
        train_data = [remaining_data[i] for i in range(len(remaining_data)) if i not in val_indices]
    else:
        print(f"  Warning: only {len(remaining_data)} samples left (< val_count={val_count}); "
              f"using 20% as validation set")
        train_data, val_data = train_test_split(remaining_data, test_size=0.2, random_state=random_state)

    print("\nData split result:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")

    if len(test_data) == 0:
        print("\n  Warning: No matching test samples found!")
        print("  Suggestion: increase tolerance or check target conductivity values")
    
    return train_data, val_data, test_data
