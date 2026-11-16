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
    cd src/downstream
    python models/train_dcr_dual_temperature.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

_PAPER_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR = _PAPER_ROOT / "models"
for p in (_PAPER_ROOT, _MODELS_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from common import split_data_by_conductivity  # noqa: E402
from dcr_dual_temperature_predictor import (  # noqa: E402
    DCRDualDataset,
    DCRDualTemperaturePredictor,
    apply_fold_normalization,
    build_dcr_samples_from_component_csv,
    create_random_backbone_for_dcr,
    load_pretrained_backbone_for_dcr,
    normalize_sample,
)

def _default_pretrain_path() -> str:
    p = _REPO_ROOT / "data" / "checkpoints" / "best_model.pt"
    if p.is_file():
        return str(p)
    return str(_MODELS_DIR / "outputs" / "pretrain_rdf_best_model.pth")

def _eval_joint_metrics(
    model: DCRDualTemperaturePredictor,
    loader: DataLoader,
    device: torch.device,
    y_std_t: torch.Tensor,
    y_mean_t: torch.Tensor,
) -> tuple[float, float, float]:

    model.eval()
    se = 0.0
    n = 0
    ae0 = 0.0
    ae1 = 0.0
    n_rows = 0
    with torch.no_grad():
        for batch in loader:
            mf = batch["mol_fractions"].to(device)
            mc = batch["mol_counts"].to(device)
            ip = batch["ion_pair_ids"].to(device)
            rp = batch["r_positions"].to(device)
            mk = batch["mask"].to(device)
            y = batch["y_dcr"].to(device)
            pv = batch["prop_vec"].to(device)
            pred = model(mf, mc, ip, rp, mk, pv)
            pred_m = pred * y_std_t + y_mean_t
            y_m = y * y_std_t + y_mean_t
            se += torch.sum((pred_m - y_m) ** 2).item()
            n += y.shape[0] * 2
            ae0 += torch.sum(torch.abs(pred_m[:, 0] - y_m[:, 0])).item()
            ae1 += torch.sum(torch.abs(pred_m[:, 1] - y_m[:, 1])).item()
            n_rows += y.shape[0]
    rmse = (se / max(n, 1)) ** 0.5
    mae0 = ae0 / max(n_rows, 1)
    mae1 = ae1 / max(n_rows, 1)
    return rmse, mae0, mae1

def train_one_fold(
    fold_idx: int,
    train_samples: list,
    val_samples: list,
    mol_columns: list,
    ion_pair_vocab: dict,
    pretrained_path: str,
    cfg: dict,
    device: torch.device,
    test_samples: list | None,
):
    tr_idx = list(range(len(train_samples)))
    p_mean, p_std, y_mean, y_std = apply_fold_normalization(train_samples, tr_idx)
    train_norm = [normalize_sample(train_samples[i], p_mean, p_std, y_mean, y_std) for i in tr_idx]
    val_norm = [normalize_sample(s, p_mean, p_std, y_mean, y_std) for s in val_samples]

    train_ds = DCRDualDataset(train_norm, mol_columns, ion_pair_vocab)
    val_ds = DCRDualDataset(val_norm, mol_columns, ion_pair_vocab)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    if cfg.get("from_scratch"):
        backbone = create_random_backbone_for_dcr(pretrained_path, mol_columns, device)
    else:
        backbone = load_pretrained_backbone_for_dcr(pretrained_path, mol_columns, device)
    model = DCRDualTemperaturePredictor(
        pretrained_model=backbone,
        num_components=len(mol_columns),
        mol_columns=mol_columns,
        freeze_encoder=cfg["freeze_encoder"],
        composition_dim=cfg["composition_dim"],
        regression_hidden_dims=cfg["regression_hidden_dims"],
        use_prop_features=cfg.get("use_prop_features", True),
    ).to(device)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    best_val = float("inf")
    patience = 0
    best_state = None
    y_std_t = torch.tensor(y_std, device=device, dtype=torch.float32)
    y_mean_t = torch.tensor(y_mean, device=device, dtype=torch.float32)

    for epoch in range(cfg["num_epochs"]):
        if epoch == cfg.get("lr_decay_epoch", 999999):
            for g in opt.param_groups:
                g["lr"] *= cfg.get("lr_decay_factor", 0.5)

        model.train()
        for batch in train_loader:
            mf = batch["mol_fractions"].to(device)
            mc = batch["mol_counts"].to(device)
            ip = batch["ion_pair_ids"].to(device)
            rp = batch["r_positions"].to(device)
            mk = batch["mask"].to(device)
            y = batch["y_dcr"].to(device)
            pv = batch["prop_vec"].to(device)

            opt.zero_grad()
            pred = model(mf, mc, ip, rp, mk, pv)
            loss = F.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        rmse_mohm, _, _ = _eval_joint_metrics(model, val_loader, device, y_std_t, y_mean_t)

        if rmse_mohm < best_val - 1e-6:
            best_val = rmse_mohm
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if (epoch + 1) % 50 == 0:
            print(f"  fold {fold_idx+1} epoch {epoch+1} val_RMSE_mOhm(both)={rmse_mohm:.3f}")
        if patience >= cfg["early_stopping_patience"]:
            print(f"  early stop fold {fold_idx+1} at epoch {epoch+1}, best val RMSE mOhm={best_val:.3f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    meta = {
        "prop_mean": p_mean.tolist(),
        "prop_std": p_std.tolist(),
        "y_mean_mohm": y_mean.tolist(),
        "y_std_mohm": y_std.tolist(),
    }

    test_rmse = float("nan")
    test_mae_m20 = float("nan")
    test_mae_25 = float("nan")
    if test_samples:
        test_norm = [normalize_sample(s, p_mean, p_std, y_mean, y_std) for s in test_samples]
        test_loader = DataLoader(
            DCRDualDataset(test_norm, mol_columns, ion_pair_vocab),
            batch_size=cfg["batch_size"],
            shuffle=False,
        )
        test_rmse, test_mae_m20, test_mae_25 = _eval_joint_metrics(
            model, test_loader, device, y_std_t, y_mean_t
        )

    return model, meta, best_val, test_rmse, test_mae_m20, test_mae_25

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(_REPO_ROOT / "data" / "downstream" / "dcr.csv"))
    ap.add_argument("--mol-columns", type=str, default=str(_REPO_ROOT / "data" / "downstream" / "mol_columns.json"))
    ap.add_argument("--ion-pair-vocab", type=str, default=str(_REPO_ROOT / "data" / "downstream" / "ion_pair_vocab.json"))
    ap.add_argument("--pretrained", type=str, default="")
    ap.add_argument(
        "--from-scratch",
        action="store_true",
        help="randomly initialize the RDF backbone (do not load pretrained weights) and auto set freeze_encoder=False",
    )
    ap.add_argument(
        "--no-prop-features",
        action="store_true",
        help="ablation: do not concatenate conductivity/viscosity/G, regress DCR using only RDF + composition-modulated embedding",
    )
    ap.add_argument("--out-dir", type=str, default=str(_MODELS_DIR / "outputs"))
    ap.add_argument("--num-folds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-epochs", type=int, default=800)
    ap.add_argument("--early-stopping-patience", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lr-decay-epoch", type=int, default=400)
    ap.add_argument("--val-count", type=int, default=15, help="random validation sample count from remaining data, consistent with the conductivity script")
    ap.add_argument("--samples-per-target", type=int, default=3, help="number of tail targets chosen into the test set per target conductivity")
    ap.add_argument("--tolerance", type=float, default=1e-5, help="conductivity matching tolerance mS/cm, consistent with the conductivity script")
    ap.add_argument(
        "--conductivity-col-for-split",
        type=str,
        default="conductivity_mS_cm_mean5fold",
        help="conductivity column used for split_data_by_conductivity (same role as the 'conductivity' column in the conductivity script)",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    with open(args.mol_columns, encoding="utf-8") as f:
        raw = json.load(f)
    mol_columns = list(raw["mol_columns"] if isinstance(raw, dict) else raw)
    with open(args.ion_pair_vocab, encoding="utf-8") as f:
        ion_pair_vocab = json.load(f)

    samples = build_dcr_samples_from_component_csv(args.csv, mol_columns)
    print(f"Full-table sample count: {len(samples)}")

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    if args.conductivity_col_for_split not in df.columns:
        raise ValueError(f"Split column does not exist: {args.conductivity_col_for_split}")
    target_conductivities = df[args.conductivity_col_for_split].tail(10).values

    train_samples, val_pool_samples, test_samples = split_data_by_conductivity(
        samples,
        target_conductivities=target_conductivities,
        val_count=args.val_count,
        samples_per_target=args.samples_per_target,
        tolerance=args.tolerance,
        random_state=42,
    )
    print(
        f"Split result (consistent with conductivity/viscosity): train_kfold_pool={len(train_samples)}, "
        f"val_pool_unused_in_kfold={len(val_pool_samples)}, test={len(test_samples)}"
    )

    pretrained = args.pretrained or _default_pretrain_path()
    if not os.path.isfile(pretrained):
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained}")

    cfg = {
        "num_folds": args.num_folds,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "lr_decay_epoch": args.lr_decay_epoch,
        "lr_decay_factor": 0.5,
        "from_scratch": bool(args.from_scratch),
        "use_prop_features": not bool(args.no_prop_features),
        "freeze_encoder": False if args.from_scratch else True,
        "composition_dim": 128,
        "regression_hidden_dims": [256, 128, 64],
        "split": {
            "strategy": "split_data_by_conductivity_then_kfold_train_only",
            "conductivity_col_for_split": args.conductivity_col_for_split,
            "tail_n_targets": 10,
            "val_count": args.val_count,
            "samples_per_target": args.samples_per_target,
            "tolerance": args.tolerance,
            "random_state_split": 42,
            "kfold_random_state": 42,
            "n_train_kfold": len(train_samples),
            "n_val_pool": len(val_pool_samples),
            "n_test": len(test_samples),
        },
    }

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_val_scores = []
    fold_test_scores = []
    indices = np.arange(len(train_samples))

    for fold_idx, (tr, va) in enumerate(kf.split(indices)):
        tr_list = [train_samples[i] for i in tr]
        va_list = [train_samples[i] for i in va]
        model, meta, best_rmse, test_rmse, test_mae_m20, test_mae_25 = train_one_fold(
            fold_idx,
            tr_list,
            va_list,
            mol_columns,
            ion_pair_vocab,
            pretrained,
            cfg,
            device,
            test_samples,
        )
        fold_val_scores.append(best_rmse)
        fold_test_scores.append(
            {"rmse_joint_mohm": test_rmse, "mae_minus20_mohm": test_mae_m20, "mae_25_mohm": test_mae_25}
        )
        path = out_dir / f"dcr_dual_temperature_fold{fold_idx + 1}.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "norm_meta": meta,
                "fold_idx": fold_idx,
                "val_rmse_mohm_joint": best_rmse,
                "test_rmse_mohm_joint": test_rmse,
                "test_mae_mohm_minus20": test_mae_m20,
                "test_mae_mohm_25": test_mae_25,
            },
            path,
        )
        print(
            f"saved {path} | val_RMSE_joint={best_rmse:.17f} | "
            f"test_RMSE_joint={test_rmse:.17f} test_MAE_m20={test_mae_m20:.17f} test_MAE_25={test_mae_25:.17f}"
        )

    print(f"\nPer-fold validation RMSE (mOhm, within-KFold validation, two channels merged): {fold_val_scores}")
    print(f"mean={float(np.mean(fold_val_scores)):.17f} std={float(np.std(fold_val_scores, ddof=0)):.17f}")
    trmse = [x["rmse_joint_mohm"] for x in fold_test_scores]
    print(f"\nPer-fold RMSE_joint on the same test set (10 samples): {trmse}")
    print(f"test_RMSE mean={float(np.mean(trmse)):.17f} std={float(np.std(trmse, ddof=0)):.17f}")

if __name__ == "__main__":
    main()
