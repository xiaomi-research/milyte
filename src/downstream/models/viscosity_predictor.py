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
Viscosity prediction model 


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    COMPONENT_PROPERTIES,
    PROPERTY_NAMES,
    extract_physicochemical_features,
    ConductivityDataset,
    split_data_by_conductivity,
    weighted_mse_loss,
    relative_error_loss,
    hybrid_loss,
    ArrheniusTemperatureEmbedding,
    EnhancedCompositionFeatureExtractor,
    FeatureModulationFusion,
    RegressionHead,
    ensemble_metrics_with_loo_stderr,
)
from rdf_backbone import RDFPretrainModel, build_component_graphs_from_rdkit

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ConductivityPredictorEnhanced(nn.Module):

    def __init__(self, pretrained_model, num_components, mol_columns,
                 freeze_encoder=True, composition_dim=128, regression_hidden_dims=[256, 128, 64],
                 default_temperature_k: float = 298.15, arrhenius_num_ea: int = 50):
        super().__init__()
        
        self.pretrained_model = pretrained_model
        self.default_temperature_k = float(default_temperature_k)
        
        if freeze_encoder:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        self.composition_extractor = EnhancedCompositionFeatureExtractor(
            num_components=num_components,
            mol_columns=mol_columns,
            output_dim=composition_dim
        )

        structure_dim = pretrained_model.config['d_model']
        self.feature_modulation = FeatureModulationFusion(
            structure_dim=structure_dim,
            composition_dim=composition_dim
        )

        self.arrhenius_temperature = ArrheniusTemperatureEmbedding(num_ea=arrhenius_num_ea)
        self._temperature_feat_dim = arrhenius_num_ea

        self.regression_head = RegressionHead(
            input_dim=structure_dim + self._temperature_feat_dim,
            hidden_dims=regression_hidden_dims,
            dropout=0.2
        )
    
    def _temperature_batch(self, temperature_k, batch_size: int, device, dtype):

        if temperature_k is None:
            return torch.full(
                (batch_size,), self.default_temperature_k, device=device, dtype=dtype
            )
        t = torch.as_tensor(temperature_k, device=device, dtype=dtype)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        t = t.reshape(-1)
        if t.numel() == 1:
            return t.expand(batch_size)
        if t.shape[0] != batch_size:
            raise ValueError(
                f"temperature_k batch size {t.shape[0]} does not match mol_fractions batch {batch_size}"
            )
        return t
    
    def forward(self, mol_fractions, mol_counts, ion_pair_ids, r_positions, mask=None,
                temperature_k=None):

        batch_size = mol_fractions.shape[0]
        device = mol_fractions.device
        dtype = mol_fractions.dtype

        encoder_frozen = all(not p.requires_grad for p in self.pretrained_model.parameters())
        grad_through_pretrained = (not encoder_frozen) or mol_fractions.requires_grad
        with torch.set_grad_enabled(grad_through_pretrained):
            predictions, E_structure = self.pretrained_model(mol_fractions, ion_pair_ids, r_positions, mask)

        E_composition = self.composition_extractor(mol_fractions, mol_counts)

        E_modulated = self.feature_modulation(E_structure, E_composition)

        T_vec = self._temperature_batch(temperature_k, batch_size, device, dtype)
        E_temperature = self.arrhenius_temperature(T_vec)
        E_regress_in = torch.cat([E_modulated, E_temperature], dim=-1)

        conductivity_pred = self.regression_head(E_regress_in)
        
        return conductivity_pred, E_structure, E_composition, E_modulated
    
    def unfreeze_encoder(self, num_layers=2):

        if hasattr(self.pretrained_model, 'rdf_transformer'):
            transformer_layers = list(self.pretrained_model.rdf_transformer.transformer_encoder.layers)
            for layer in transformer_layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"✓ Unfroze the last {num_layers} Transformer layers")

def load_csv_data(csv_path, mol_columns):

    df = pd.read_csv(csv_path)

    conductivity_col = None
    for col in ['μ']:
        if col in df.columns:
            conductivity_col = col
            break
    
    if conductivity_col is None:
        raise ValueError(f"Viscosity column not found, available columns: {df.columns.tolist()}")
    
    samples = []
    for _, row in df.iterrows():

        atom_counts = {}
        
        for col in mol_columns:
            comp_name = col.replace('mol_count_', '')

            if comp_name in row:
                atom_counts[comp_name.lower()] = float(row[comp_name])
            elif col in row:
                atom_counts[comp_name.lower()] = float(row[col])

        ratio_str = '_'.join([f"{k}:{v:.3f}" for k, v in sorted(atom_counts.items())])
        
        sample = {
            'atom_counts': atom_counts,
            'conductivity': float(row[conductivity_col]),
            'ratio_str': ratio_str
        }
        
        samples.append(sample)
    
    return samples

def train_single_fold(fold_idx, train_samples, val_samples, mol_columns, ion_pair_vocab, 
                      pretrained_model_path, train_config, device):

    print(f"\n{'='*60}")
    print(f"Training Fold {fold_idx + 1}/{train_config['num_folds']}")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    train_dataset = ConductivityDataset(train_samples, mol_columns, ion_pair_vocab)
    val_dataset = ConductivityDataset(val_samples, mol_columns, ion_pair_vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)

    checkpoint = torch.load(pretrained_model_path, map_location=device, weights_only=False)
    
    if 'config' in checkpoint:
        pretrained_config = checkpoint['config']
    else:
        pretrained_config = {
            'd_model': 256,
            'nhead': 8,
            'num_encoder_layers': 4,
            'num_decoder_layers': 4,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'max_seq_len': 512
        }
    
    component_graphs = None
    if pretrained_config.get('use_se3_equivariance', False):
        component_graphs = build_component_graphs_from_rdkit(
            mol_columns,
            cutoff=float(pretrained_config.get('se3_cutoff', 6.0)),
            seed=int(pretrained_config.get('rdkit_seed', 0)),
        )
        print("✓ Built SE(3) component 3D graphs (RDKit) on downstream load")
    pretrained_model = RDFPretrainModel(pretrained_config, component_graphs=component_graphs).to(device)
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])

    num_components = len(mol_columns)
    model = ConductivityPredictorEnhanced(
        pretrained_model=pretrained_model,
        num_components=num_components,
        mol_columns=mol_columns,
        freeze_encoder=train_config['freeze_encoder'],
        composition_dim=train_config['composition_dim'],
        regression_hidden_dims=train_config['regression_hidden_dims'],
        default_temperature_k=float(train_config.get('default_temperature_k', 298.15)),
        arrhenius_num_ea=int(train_config.get('arrhenius_num_ea', 50)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )

    best_val_mae = float('inf')
    patience_counter = 0
    best_model_state = None
    train_history = {'train_mae': [], 'val_mae': [], 'val_r2': []}
    
    for epoch in range(train_config['num_epochs']):

        if epoch == train_config['lr_decay_epoch']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= train_config['lr_decay_factor']
            print(f"Learning rate decayed to: {param_group['lr']:.2e}")

        model.train()
        train_loss_sum = 0
        train_preds_all = []
        train_targets_all = []
        
        for batch in train_loader:
            mol_fractions = batch['mol_fractions'].to(device)
            mol_counts = batch['mol_counts'].to(device)
            ion_pair_ids = batch['ion_pair_ids'].to(device)
            r_positions = batch['r_positions'].to(device)
            mask = batch['mask'].to(device)
            conductivity = batch['conductivity'].to(device).squeeze(-1)
            
            optimizer.zero_grad()
            
            pred_conductivity, _, _, _ = model(
                mol_fractions, mol_counts, ion_pair_ids, r_positions, mask
            )

            if train_config['loss_type'] == 'mse':
                loss = F.mse_loss(pred_conductivity, conductivity)
            elif train_config['loss_type'] == 'weighted_mse':
                loss = weighted_mse_loss(pred_conductivity, conductivity, weight_type='dynamic')
            elif train_config['loss_type'] == 'relative':
                loss = relative_error_loss(pred_conductivity, conductivity)
            elif train_config['loss_type'] == 'hybrid':
                loss = hybrid_loss(pred_conductivity, conductivity, alpha=0.5, beta=0.5, weight_type='dynamic')
            else:
                loss = F.mse_loss(pred_conductivity, conductivity)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_preds_all.extend(pred_conductivity.detach().cpu().numpy())
            train_targets_all.extend(conductivity.cpu().numpy())
        
        train_mae = mean_absolute_error(train_targets_all, train_preds_all)

        model.eval()
        val_loss_sum = 0
        val_preds_all = []
        val_targets_all = []
        
        with torch.no_grad():
            for batch in val_loader:
                mol_fractions = batch['mol_fractions'].to(device)
                mol_counts = batch['mol_counts'].to(device)
                ion_pair_ids = batch['ion_pair_ids'].to(device)
                r_positions = batch['r_positions'].to(device)
                mask = batch['mask'].to(device)
                conductivity = batch['conductivity'].to(device).squeeze(-1)
                
                pred_conductivity, _, _, _ = model(
                    mol_fractions, mol_counts, ion_pair_ids, r_positions, mask
                )
                
                loss = F.mse_loss(pred_conductivity, conductivity)
                val_loss_sum += loss.item()
                val_preds_all.extend(pred_conductivity.cpu().numpy())
                val_targets_all.extend(conductivity.cpu().numpy())
        
        val_mae = mean_absolute_error(val_targets_all, val_preds_all)
        val_r2 = r2_score(val_targets_all, val_preds_all)
        
        train_history['train_mae'].append(train_mae)
        train_history['val_mae'].append(val_mae)
        train_history['val_r2'].append(val_r2)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{train_config['num_epochs']}: "
                  f"Train MAE={train_mae:.4f}, Val MAE={val_mae:.4f}, Val R²={val_r2:.4f}")
        
        if patience_counter >= train_config['early_stopping_patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    return model, train_history, best_val_mae

def evaluate_cv(models, dataloader, device):

    all_preds = []
    all_targets = []
    
    for model in models:
        model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            mol_fractions = batch['mol_fractions'].to(device)
            mol_counts = batch['mol_counts'].to(device)
            ion_pair_ids = batch['ion_pair_ids'].to(device)
            r_positions = batch['r_positions'].to(device)
            mask = batch['mask'].to(device)
            conductivity = batch['conductivity'].to(device).squeeze(-1)

            ensemble_preds = []
            for model in models:
                pred_conductivity, _, _, _ = model(
                    mol_fractions, mol_counts, ion_pair_ids, r_positions, mask
                )
                ensemble_preds.append(pred_conductivity.cpu().numpy())

            avg_pred = np.mean(ensemble_preds, axis=0)
            
            all_preds.extend(avg_pred)
            all_targets.extend(conductivity.cpu().numpy())
    
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    
    return mae, r2, rmse, all_preds, all_targets

def stacked_test_predictions_viscosity(models, dataloader, device):

    k = len(models)
    per_model_blocks: list[list] = [[] for _ in range(k)]
    target_blocks: list = []
    for m in models:
        m.eval()
    with torch.no_grad():
        for batch in dataloader:
            mol_fractions = batch["mol_fractions"].to(device)
            mol_counts = batch["mol_counts"].to(device)
            ion_pair_ids = batch["ion_pair_ids"].to(device)
            r_positions = batch["r_positions"].to(device)
            mask = batch["mask"].to(device)
            conductivity = batch["conductivity"].to(device).squeeze(-1)
            target_blocks.append(conductivity.detach().float().cpu().numpy().reshape(-1))
            for mi, model in enumerate(models):
                pred_conductivity, _, _, _ = model(
                    mol_fractions, mol_counts, ion_pair_ids, r_positions, mask
                )
                per_model_blocks[mi].append(pred_conductivity.detach().float().cpu().numpy().reshape(-1))
    y_true = np.concatenate(target_blocks)
    preds_stack = np.stack([np.concatenate(per_model_blocks[mi]) for mi in range(k)], axis=0)
    return y_true, preds_stack

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    models_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(os.path.dirname(base_dir))
    data_dir = os.path.join(repo_root, 'data', 'downstream')
    output_dir = os.path.join(models_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    train_config = {
        'num_folds': 5,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 10000,
        'early_stopping_patience': 300,
        'lr_decay_epoch': 250,
        'lr_decay_factor': 0.5,

        'freeze_encoder': True,
        'composition_dim': 128,
        'regression_hidden_dims': [128, 64],

        'default_temperature_k': 298.15,
        'arrhenius_num_ea': 50,

        'loss_type': 'hybrid'
    }

    print("\nLoading data...")
    csv_path = os.path.join(data_dir, 'viscosity.csv')
    mol_columns_path = os.path.join(data_dir, 'mol_columns.json')
    ion_pair_vocab_path = os.path.join(data_dir, 'ion_pair_vocab.json')

    pretrained_model_path = os.path.join(
        repo_root, 'data', 'checkpoints', 'best_model.pt'
    )
    if not os.path.exists(pretrained_model_path):
        pretrained_model_path = os.path.join(output_dir, 'pretrain_rdf_best_model.pth')
        print(f"⚠️ Pretrained model not found, using: {pretrained_model_path}")
    else:
        print(f"✓ Using pretrained model: {pretrained_model_path}")
    
    with open(mol_columns_path, 'r') as f:
        mol_columns_data = json.load(f)

        if isinstance(mol_columns_data, dict):
            mol_columns = mol_columns_data['mol_columns']
        else:
            mol_columns = mol_columns_data
    
    with open(ion_pair_vocab_path, 'r') as f:
        ion_pair_vocab = json.load(f)

    all_samples = load_csv_data(csv_path, mol_columns)

    df = pd.read_csv(csv_path)

    conductivity_col = 'μ'
    target_conductivities = df[conductivity_col].tail(10).values

    train_samples, val_samples, test_samples = split_data_by_conductivity(
        all_samples,
        target_conductivities=target_conductivities,
        val_count=15,
        samples_per_target=3,
        tolerance=0.00001,
        random_state=0
    )
    
    print("\nData split:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    print(f"  Test: {len(test_samples)} samples")

    kfold = KFold(n_splits=train_config['num_folds'], shuffle=True, random_state=42)
    
    fold_models = []
    fold_val_maes = []
    fold_test_results = []
    
    for fold_idx, (train_idx, val_fold_idx) in enumerate(kfold.split(train_samples)):
        fold_train_samples = [train_samples[i] for i in train_idx]
        fold_val_samples = [train_samples[i] for i in val_fold_idx]

        model, history, val_mae = train_single_fold(
            fold_idx, fold_train_samples, fold_val_samples,
            mol_columns, ion_pair_vocab, pretrained_model_path,
            train_config, device
        )
        
        fold_models.append(model)
        fold_val_maes.append(val_mae)

        test_dataset = ConductivityDataset(test_samples, mol_columns, ion_pair_vocab)
        test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False)
        
        test_mae, test_r2, test_rmse, _, _ = evaluate_cv([model], test_loader, device)
        fold_test_results.append({'mae': test_mae, 'r2': test_r2, 'rmse': test_rmse})
        
        print(f"\nFold {fold_idx + 1} test results:")
        print(f"  MAE: {test_mae:.4f} mPa·s")
        print(f"  R²: {test_r2:.4f}")
        print(f"  RMSE: {test_rmse:.4f} mPa·s")

    print(f"\n{'='*70}")
    print("Cross-validation summary (enhanced physicochemical feature fusion)")
    print(f"{'='*70}")

    print("Validation split stats:")
    print(f"  Mean MAE: {np.mean(fold_val_maes):.2f} ± {np.std(fold_val_maes):.2f} mPa·s")

    print("\nTest split stats:")
    test_maes = [r['mae'] for r in fold_test_results]
    test_r2s = [r['r2'] for r in fold_test_results]
    print(f"  Mean MAE: {np.mean(test_maes):.2f} ± {np.std(test_maes):.2f} mPa·s")
    print(f"  Mean R²: {np.mean(test_r2s):.4f} ± {np.std(test_r2s):.4f}")

    test_dataset = ConductivityDataset(test_samples, mol_columns, ion_pair_vocab)
    test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False)
    
    y_te, pred_stack = stacked_test_predictions_viscosity(fold_models, test_loader, device)
    ens_err = ensemble_metrics_with_loo_stderr(y_te, pred_stack)

    print(f"\nEnsemble prediction ({train_config['num_folds']} model average):")
    print(
        f"  MAE: {ens_err['mae']:.2f} ± {ens_err['mae_stderr']:.2f} mPa·s "
        f"(error bar: std of leave-one-model-out sub-ensemble MAE)"
    )
    print(f"  R²: {ens_err['r2']:.4f} ± {ens_err['r2_stderr']:.4f}")
    print(f"  RMSE: {ens_err['rmse']:.2f} ± {ens_err['rmse_stderr']:.2f} mPa·s")

    for idx, model in enumerate(fold_models):
        save_path = os.path.join(output_dir, f'viscosity_enhanced_fold{idx+1}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': train_config,
            'fold_idx': idx,
            'val_mae': fold_val_maes[idx],
            'test_results': fold_test_results[idx]
        }, save_path)
    
    print(f"\nModel saved to: {output_dir}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
