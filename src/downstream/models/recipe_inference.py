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
"""Batch inference: electrolyte recipe -> conductivity / viscosity / free energy.

This module is self-contained (it does not depend on plotting code).  It loads
the released RDF backbone and the 5-fold downstream checkpoints produced by the
training scripts, then writes predictions for every row of an input CSV.

Example (from the repository root):

    python src/downstream/models/recipe_inference.py \
      --csv data/downstream/conductivity.csv \
      --out /tmp/predictions.csv

The input CSV must contain one column per component (``EC``, ``EMC``, ``LiPF6``,
...).  Missing components are treated as zero.  A ``valeronitrile`` column is
accepted as an alias for ``VN``.

Checkpoint layout expected under ``--models-output`` (default:
``src/downstream/models/outputs``)::

    conductivity_enhanced_fold{1..5}.pth
    viscosity_enhanced_fold{1..5}.pth
    free_energy_enhanced_fold{1..5}.pth

These files are produced by the corresponding training scripts and are
distributed separately via Hugging Face (see README).  To run inference with held-out/private folds,
point ``--models-output`` at that directory and adjust ``--num-folds``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_THIS_FILE = Path(__file__).resolve()
_MODELS_DIR = _THIS_FILE.parent
_PAPER_ROOT = _MODELS_DIR.parent          # src/downstream
_REPO_ROOT = _PAPER_ROOT.parent.parent    # repository root
for p in (_PAPER_ROOT, _MODELS_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from common import ConductivityDataset  # noqa: E402
from conductivity_predictor import ConductivityPredictorEnhanced  # noqa: E402
from free_energy_predictor import (  # noqa: E402
    FreeEnergyDataset,
    FreeEnergyPredictorEnhanced,
    mol_column_to_coord_column,
)
from rdf_backbone import (  # noqa: E402
    RDFPretrainModel,
    build_component_graphs_from_rdkit,
)
from viscosity_predictor import (  # noqa: E402
    ConductivityPredictorEnhanced as ViscosityPredictorEnhanced,
)

_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "downstream"
_DEFAULT_MODELS_OUTPUT = _MODELS_DIR / "outputs"
_DEFAULT_PRETRAINED = _REPO_ROOT / "data" / "checkpoints" / "best_model.pt"


# ---------------------------------------------------------------------------
# JSON / dataframe helpers
# ---------------------------------------------------------------------------
def load_mol_columns_json(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw: Any = json.load(f)
    if isinstance(raw, dict) and "mol_columns" in raw:
        return list(raw["mol_columns"])
    return list(raw)


def load_ion_pair_vocab(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(v):
        return 0.0
    return v


def row_value_for_component(
    row: pd.Series,
    comp_name: str,
    column_aliases: Mapping[str, str],
) -> float:
    target = comp_name.strip()
    for c in row.index:
        key = str(c).strip()
        if not key:
            continue
        if key.lower() == target.lower():
            return _safe_float(row[c])
    for alt_col, tgt in column_aliases.items():
        if str(tgt).strip().lower() != target.lower():
            continue
        if alt_col in row.index:
            return _safe_float(row[alt_col])
    return 0.0


def dataframe_to_conductivity_samples(
    df: pd.DataFrame,
    mol_columns: list[str],
    column_aliases: Mapping[str, str] | None = None,
    dummy_conductivity: float = 0.0,
) -> list[dict]:
    aliases = column_aliases or {}
    samples: list[dict] = []
    for _, row in df.iterrows():
        atom_counts: dict[str, float] = {}
        for mc in mol_columns:
            comp = mc.replace("mol_count_", "")
            atom_counts[comp.lower()] = row_value_for_component(row, comp, aliases)
        ratio_str = "_".join(f"{k}:{v:.6f}" for k, v in sorted(atom_counts.items()))
        samples.append(
            {
                "atom_counts": atom_counts,
                "conductivity": float(dummy_conductivity),
                "ratio_str": ratio_str,
            }
        )
    return samples


def dataframe_to_free_energy_samples(
    df: pd.DataFrame,
    mol_columns: list[str],
    column_aliases: Mapping[str, str] | None = None,
    dummy_g: float = 0.0,
    coordination_default: float = 0.0,
) -> list[dict]:
    aliases = column_aliases or {}
    z = float(coordination_default)
    samples: list[dict] = []
    for _, row in df.iterrows():
        atom_counts: dict[str, float] = {}
        for mc in mol_columns:
            comp = mc.replace("mol_count_", "")
            atom_counts[comp.lower()] = row_value_for_component(row, comp, aliases)
        coord_list: list[float] = []
        for mc in mol_columns:
            cc = mol_column_to_coord_column(mc)
            if cc in row.index:
                coord_list.append(_safe_float(row[cc]))
            else:
                coord_list.append(z)
        ratio_str = "_".join(f"{k}:{v:.6f}" for k, v in sorted(atom_counts.items()))
        samples.append(
            {
                "atom_counts": atom_counts,
                "coordination_n": coord_list,
                "free_energy": float(dummy_g),
                "ratio_str": ratio_str,
            }
        )
    return samples


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_pretrained_backbone(
    pretrained_path: str | Path,
    mol_columns: list[str],
    device: torch.device,
) -> tuple[RDFPretrainModel, dict]:
    pretrained_path = Path(pretrained_path)
    if not pretrained_path.is_file():
        raise FileNotFoundError(
            f"Pretrained RDF backbone not found: {pretrained_path}\n"
            "Expected data/checkpoints/best_model.pt, or pass --pretrained."
        )
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)

    pretrained_config = dict(checkpoint.get("config", {}))
    pretrained_config.setdefault("d_model", 256)
    pretrained_config.setdefault("nhead", 8)
    pretrained_config.setdefault("num_layers", 5)
    pretrained_config.setdefault("dim_feedforward", 768)
    pretrained_config.setdefault("dropout", 0.15)
    pretrained_config.setdefault("max_seq_len", 1200)
    pretrained_config.setdefault("component_hidden_dim", 128)
    pretrained_config.setdefault("num_ion_pairs", 20)
    pretrained_config["num_components"] = len(mol_columns)

    component_graphs = None
    if pretrained_config.get("use_se3_equivariance", False):
        component_graphs = build_component_graphs_from_rdkit(
            mol_columns,
            cutoff=float(pretrained_config.get("se3_cutoff", 6.0)),
            seed=int(pretrained_config.get("rdkit_seed", 0)),
        )

    model = RDFPretrainModel(pretrained_config, component_graphs=component_graphs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, pretrained_config


def _load_fold_checkpoint(path: str | Path, device: torch.device) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Downstream checkpoint not found: {path}\n"
            "Train the downstream models first (see README) or pass --models-output."
        )
    return torch.load(path, map_location=device, weights_only=False)


def load_conductivity_fold_model(
    ckpt_path: str,
    mol_columns: list,
    ion_pair_vocab: dict,
    pretrained_model_path: str,
    device: torch.device,
) -> ConductivityPredictorEnhanced:
    ckpt = _load_fold_checkpoint(ckpt_path, device)
    train_config = ckpt.get("config", {})
    pretrained_model, _ = _load_pretrained_backbone(pretrained_model_path, mol_columns, device)
    model = ConductivityPredictorEnhanced(
        pretrained_model=pretrained_model,
        num_components=len(mol_columns),
        mol_columns=mol_columns,
        freeze_encoder=bool(train_config.get("freeze_encoder", False)),
        composition_dim=int(train_config.get("composition_dim", 128)),
        regression_hidden_dims=train_config.get("regression_hidden_dims", [256, 128, 64]),
        default_temperature_k=float(train_config.get("default_temperature_k", 298.15)),
        arrhenius_num_ea=int(train_config.get("arrhenius_num_ea", 50)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_viscosity_fold_model(
    ckpt_path: str,
    mol_columns: list,
    ion_pair_vocab: dict,
    pretrained_model_path: str,
    device: torch.device,
) -> ViscosityPredictorEnhanced:
    ckpt = _load_fold_checkpoint(ckpt_path, device)
    train_config = ckpt.get("config", {})
    pretrained_model, _ = _load_pretrained_backbone(pretrained_model_path, mol_columns, device)
    model = ViscosityPredictorEnhanced(
        pretrained_model=pretrained_model,
        num_components=len(mol_columns),
        mol_columns=mol_columns,
        freeze_encoder=bool(train_config.get("freeze_encoder", True)),
        composition_dim=int(train_config.get("composition_dim", 128)),
        regression_hidden_dims=train_config.get("regression_hidden_dims", [128, 64]),
        default_temperature_k=float(train_config.get("default_temperature_k", 298.15)),
        arrhenius_num_ea=int(train_config.get("arrhenius_num_ea", 50)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_free_energy_fold_model(
    ckpt_path: str,
    mol_columns: list,
    ion_pair_vocab: dict,
    pretrained_model_path: str,
    device: torch.device,
) -> FreeEnergyPredictorEnhanced:
    ckpt = _load_fold_checkpoint(ckpt_path, device)
    train_config = ckpt.get("config", {})
    pretrained_model, _ = _load_pretrained_backbone(pretrained_model_path, mol_columns, device)
    model = FreeEnergyPredictorEnhanced(
        pretrained_model=pretrained_model,
        num_components=len(mol_columns),
        mol_columns=mol_columns,
        freeze_encoder=bool(train_config.get("freeze_encoder", True)),
        composition_dim=int(train_config.get("composition_dim", 128)),
        coord_embed_dim=int(train_config.get("coord_embed_dim", 64)),
        coord_hidden_dim=int(train_config.get("coord_hidden_dim", 128)),
        regression_hidden_dims=train_config.get("regression_hidden_dims", [256, 128, 64]),
        regress_proj_dim=train_config.get("regress_proj_dim"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@torch.no_grad()
def _predict_conductivity_like(
    model: torch.nn.Module,
    samples: list[dict],
    mol_columns: list[str],
    ion_pair_vocab: dict,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    ds = ConductivityDataset(samples, mol_columns, ion_pair_vocab)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    for batch in loader:
        mol_fractions = batch["mol_fractions"].to(device)
        mol_counts = batch["mol_counts"].to(device)
        ion_pair_ids = batch["ion_pair_ids"].to(device)
        r_positions = batch["r_positions"].to(device)
        mask = batch["mask"].to(device)
        pred, _, _, _ = model(mol_fractions, mol_counts, ion_pair_ids, r_positions, mask)
        preds.append(pred.detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32)


@torch.no_grad()
def _predict_free_energy_like(
    model: torch.nn.Module,
    samples: list[dict],
    mol_columns: list[str],
    ion_pair_vocab: dict,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    ds = FreeEnergyDataset(samples, mol_columns, ion_pair_vocab)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    for batch in loader:
        mol_fractions = batch["mol_fractions"].to(device)
        mol_counts = batch["mol_counts"].to(device)
        ion_pair_ids = batch["ion_pair_ids"].to(device)
        r_positions = batch["r_positions"].to(device)
        mask = batch["mask"].to(device)
        coordination_n = batch["coordination_n"].to(device)
        pred, _, _, _ = model(
            mol_fractions, mol_counts, ion_pair_ids, r_positions, coordination_n, mask
        )
        preds.append(pred.detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32)


def ensemble_mean_predict(
    *,
    task: str,
    samples_cond_style: list[dict],
    samples_fe_style: list[dict],
    mol_columns: list[str],
    ion_pair_vocab: dict,
    pretrained_path: str,
    models_output: str,
    device: torch.device,
    batch_size: int,
    num_folds: int = 5,
) -> np.ndarray:
    if task not in {"conductivity", "viscosity", "free_energy"}:
        raise ValueError(f"Unknown task: {task}")

    samples = samples_fe_style if task == "free_energy" else samples_cond_style
    acc = np.zeros(len(samples), dtype=np.float64)
    for fold in range(1, num_folds + 1):
        if task == "conductivity":
            ckpt = os.path.join(models_output, f"conductivity_enhanced_fold{fold}.pth")
            model = load_conductivity_fold_model(
                ckpt, mol_columns, ion_pair_vocab, pretrained_path, device
            )
            acc += _predict_conductivity_like(
                model, samples, mol_columns, ion_pair_vocab, device, batch_size
            )
        elif task == "viscosity":
            ckpt = os.path.join(models_output, f"viscosity_enhanced_fold{fold}.pth")
            model = load_viscosity_fold_model(
                ckpt, mol_columns, ion_pair_vocab, pretrained_path, device
            )
            acc += _predict_conductivity_like(
                model, samples, mol_columns, ion_pair_vocab, device, batch_size
            )
        else:
            ckpt = os.path.join(models_output, f"free_energy_enhanced_fold{fold}.pth")
            model = load_free_energy_fold_model(
                ckpt, mol_columns, ion_pair_vocab, pretrained_path, device
            )
            acc += _predict_free_energy_like(
                model, samples, mol_columns, ion_pair_vocab, device, batch_size
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return acc / float(num_folds)


def predict_properties_for_dataframe(
    df: pd.DataFrame,
    *,
    mol_columns: list[str],
    ion_pair_vocab: dict,
    pretrained_path: str,
    models_output: str,
    column_aliases: Mapping[str, str] | None = None,
    device: torch.device | None = None,
    batch_size: int = 16,
    num_folds: int = 5,
    tasks: tuple[str, ...] = ("conductivity", "viscosity", "free_energy"),
) -> dict[str, np.ndarray]:
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s_cond = dataframe_to_conductivity_samples(df, mol_columns, column_aliases=column_aliases)
    s_fe = dataframe_to_free_energy_samples(df, mol_columns, column_aliases=column_aliases)

    out: dict[str, np.ndarray] = {}
    for task in tasks:
        pred = ensemble_mean_predict(
            task=task,
            samples_cond_style=s_cond,
            samples_fe_style=s_fe,
            mol_columns=mol_columns,
            ion_pair_vocab=ion_pair_vocab,
            pretrained_path=pretrained_path,
            models_output=models_output,
            device=dev,
            batch_size=batch_size,
            num_folds=num_folds,
        )
        if task == "conductivity":
            out["pred_conductivity_ms_cm"] = pred
        elif task == "viscosity":
            out["pred_viscosity_cp"] = pred
        else:
            out["pred_G_kJ_per_mol"] = pred
    return out


def default_column_aliases() -> dict[str, str]:
    return {"valeronitrile": "VN"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch inference: recipe CSV -> conductivity / viscosity / free energy."
    )
    parser.add_argument("--csv", required=True, help="Input recipe CSV")
    parser.add_argument(
        "--out", default=None,
        help="Output CSV (default: <input stem>_predictions.csv next to the input)",
    )
    parser.add_argument("--mol-columns", default=str(_DEFAULT_DATA_DIR / "mol_columns.json"))
    parser.add_argument("--ion-pair-vocab", default=str(_DEFAULT_DATA_DIR / "ion_pair_vocab.json"))
    parser.add_argument(
        "--pretrained", default=str(_DEFAULT_PRETRAINED),
        help="RDF pretrained backbone (default: src/pretrain/checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--models-output", default=str(_DEFAULT_MODELS_OUTPUT),
        help="Directory with *_enhanced_fold{1..N}.pth checkpoints",
    )
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument(
        "--tasks", nargs="+", default=["conductivity", "viscosity", "free_energy"],
        choices=["conductivity", "viscosity", "free_energy"],
        help="Properties to predict (default: all three)",
    )
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    mol_columns = load_mol_columns_json(args.mol_columns)
    ion_pair_vocab = load_ion_pair_vocab(args.ion_pair_vocab)

    print(f"Input   : {args.csv}  ({len(df)} rows)")
    print(f"Device  : {device}")
    print(f"Folds   : {args.num_folds}  ({args.models_output})")
    print(f"Tasks   : {', '.join(args.tasks)}")

    preds = predict_properties_for_dataframe(
        df,
        mol_columns=mol_columns,
        ion_pair_vocab=ion_pair_vocab,
        pretrained_path=args.pretrained,
        models_output=args.models_output,
        column_aliases=default_column_aliases(),
        device=device,
        batch_size=args.batch_size,
        num_folds=args.num_folds,
        tasks=tuple(args.tasks),
    )

    out_df = df.copy()
    for key, values in preds.items():
        out_df[key] = values

    out_path = args.out or str(
        Path(args.csv).with_name(Path(args.csv).stem + "_predictions.csv")
    )
    out_df.to_csv(out_path, index=False)
    print(f"Wrote   : {out_path}")
    for key, values in preds.items():
        print(f"  {key}: mean={float(np.mean(values)):.4f}")


if __name__ == "__main__":
    main()
