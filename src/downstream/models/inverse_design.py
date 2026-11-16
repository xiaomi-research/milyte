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
Conductivity inverse design
Usage (run from ``src/downstream/models``):

python inverse_design.py \
  --checkpoint <path-to>/conductivity_model.pth \
  --pretrained ../../../data/checkpoints/best_model.pt \
  --mol-columns ../../../data/downstream/mol_columns.json \
  --target 15 \
  --local-refine

"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _SCRIPT_DIR.parent / "models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from conductivity_predictor import ConductivityPredictorEnhanced  # noqa: E402
from rdf_backbone import RDFPretrainModel, build_component_graphs_from_rdkit  # noqa: E402

_DATA_DIR = _SCRIPT_DIR.parents[2] / "data" / "downstream"
_DEFAULT_OUTPUT_ORDER_CSV = _DATA_DIR / "conductivity.csv"
_CONDUCTIVITY_HEADER_NAMES = frozenset(
    {"κ", "conductivity", "Conductivity", "conductivity_ms_cm", "Conductivity_mScm"}
)


def load_mol_columns(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data["mol_columns"]
    return data


def _mol_column_to_short_name(col: str) -> str:
    return col.replace("mol_count_", "").strip()


def _normalize_component_key(name: str) -> str:
    return _mol_column_to_short_name(name).lower()


def read_csv_component_column_order(csv_path: str) -> list[str]:
    """Read the order of component column names from the CSV header (excluding target columns)."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV for column order not found: {csv_path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f))
    out = []
    for raw in header:
        h = (raw or "").strip()
        if not h or h in _CONDUCTIVITY_HEADER_NAMES:
            continue
        out.append(h)
    return out


def reorder_result_to_csv_columns(
    result: dict,
    mol_columns: list,
    csv_path: str,
) -> None:
    """
    Reorder result['mol_fractions'] (ordered as mol_columns) to match the CSV header column
    order, and rewrite result['recipe'] (inserted as a dict in that order for JSON/CSV alignment).
    """
    csv_order = read_csv_component_column_order(csv_path)
    model_norm_to_idx = {_normalize_component_key(c): i for i, c in enumerate(mol_columns)}
    csv_norms = [_normalize_component_key(h) for h in csv_order]

    missing_in_csv = set(model_norm_to_idx) - set(csv_norms)
    if missing_in_csv:
        raise ValueError(
            f"CSV header missing columns corresponding to mol_columns: {sorted(missing_in_csv)} (file: {csv_path})"
        )
    extra_in_csv = set(csv_norms) - set(model_norm_to_idx)
    if extra_in_csv:
        raise ValueError(
            f"CSV header has columns not defined in mol_columns: {sorted(extra_in_csv)} (file: {csv_path})"
        )
    if len(csv_norms) != len(model_norm_to_idx):
        raise ValueError(
            f"CSV component column count {len(csv_norms)} does not match mol_columns count {len(model_norm_to_idx)}"
        )

    fr = result["mol_fractions"]
    if len(fr) != len(mol_columns):
        raise ValueError("mol_fractions length does not match mol_columns")

    model_indices = [model_norm_to_idx[k] for k in csv_norms]
    result["mol_fractions"] = [fr[j] for j in model_indices]
    # Use the original header strings as keys (matching CSV column names, e.g. LiPF6)
    result["recipe"] = {csv_order[k]: fr[model_indices[k]] for k in range(len(csv_order))}
    result["component_column_order"] = list(csv_order)
    result["output_order_source_csv"] = str(Path(csv_path).resolve())


def build_rdf_batch(
    batch_size: int,
    max_seq_len: int,
    ion_pair_id: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aligns with the RDF placeholder logic in ConductivityDataset."""
    ion_pair_ids = np.full((batch_size, max_seq_len), ion_pair_id, dtype=np.int64)
    r_positions = np.zeros((batch_size, max_seq_len), dtype=np.int64)
    r_positions[:, :seq_len] = np.arange(seq_len, dtype=np.int64)
    mask = np.ones((batch_size, max_seq_len), dtype=bool)
    mask[:, :seq_len] = False
    return (
        torch.from_numpy(ion_pair_ids).to(device),
        torch.from_numpy(r_positions).to(device),
        torch.from_numpy(mask).to(device),
    )


def load_trained_enhanced(
    checkpoint_path: str,
    pretrained_path: str,
    mol_columns: list,
    device: torch.device,
) -> tuple[ConductivityPredictorEnhanced, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_config = ckpt.get("config", {})
    state = ckpt.get("model_state_dict", {})

    pre = torch.load(pretrained_path, map_location=device, weights_only=False)
    pretrained_config = pre.get("config")
    if pretrained_config is None:
        raise ValueError(f"Pretrained checkpoint is missing config: {pretrained_path}")


    has_se3_in_finetune = any(
        k.startswith("pretrained_model.se3_composition_encoder")
        or k.startswith("pretrained_model.formula_se3_fusion")
        for k in state
    )
    pretrained_config = copy.deepcopy(pretrained_config)
    if has_se3_in_finetune:
        pretrained_config["use_se3_equivariance"] = True

    component_graphs = None
    if pretrained_config.get("use_se3_equivariance", False):
        component_graphs = build_component_graphs_from_rdkit(
            mol_columns,
            cutoff=float(pretrained_config.get("se3_cutoff", 6.0)),
            seed=int(pretrained_config.get("rdkit_seed", 0)),
        )

    pretrained_model = RDFPretrainModel(pretrained_config, component_graphs=component_graphs).to(device)
    num_components = len(mol_columns)
    model = ConductivityPredictorEnhanced(
        pretrained_model=pretrained_model,
        num_components=num_components,
        mol_columns=mol_columns,
        freeze_encoder=train_config.get("freeze_encoder", True),
        composition_dim=int(train_config.get("composition_dim", 128)),
        regression_hidden_dims=train_config.get("regression_hidden_dims", [256, 128, 64]),
        default_temperature_k=float(train_config.get("default_temperature_k", 298.15)),
        arrhenius_num_ea=int(train_config.get("arrhenius_num_ea", 50)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model, train_config


def inverse_optimize_mol_fractions(
    model: ConductivityPredictorEnhanced,
    train_config: dict,
    mol_columns: list,
    target_conductivity: float,
    device: torch.device,
    *,
    num_restarts: int = 8,
    steps: int = 500,
    lr: float = 0.05,
    mol_count_scale: float = 1.0,
    max_seq_len: int = 1200,
    seq_len: int = 100,
    ion_pair_id: int = 0,
    temperature_k: float | None = None,
    prior_weight: float = 0.0,
    init_fractions: np.ndarray | None = None,
) -> dict:
    """
    Parallel multi-restart: Adam over softmax(logits), minimizing (pred - target)^2 +
    prior_weight * MSE(f, f0).
    """
    c = len(mol_columns)
    t_k = temperature_k if temperature_k is not None else float(train_config.get("default_temperature_k", 298.15))
    t_tensor = torch.full((num_restarts,), t_k, device=device, dtype=torch.float32)

    ion_pair_ids, r_positions, mask = build_rdf_batch(
        num_restarts, max_seq_len, ion_pair_id, seq_len, device
    )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    target = torch.tensor(target_conductivity, device=device, dtype=torch.float32)

    if init_fractions is not None:
        f0 = torch.as_tensor(init_fractions, dtype=torch.float32, device=device).reshape(1, c)
        if f0.shape[1] != c:
            raise ValueError(f"init_fractions length should be {c}, got {f0.shape[1]}")
        base = torch.log(f0.clamp(min=1e-8)).expand(num_restarts, -1)
        logits = base + torch.randn(num_restarts, c, device=device, dtype=torch.float32) * 0.05
    else:
        logits = torch.randn(num_restarts, c, device=device, dtype=torch.float32) * 0.5

    logits = logits.requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)

    f0_batch = None
    if prior_weight > 0 and init_fractions is not None:
        f0_batch = f0.expand(num_restarts, -1).clone()

    best_loss = float("inf")
    best_frac: torch.Tensor | None = None
    best_pred: torch.Tensor | None = None

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        mol_fractions = F.softmax(logits, dim=-1)
        mol_counts = mol_fractions * mol_count_scale

        pred, _, _, _ = model(
            mol_fractions,
            mol_counts,
            ion_pair_ids,
            r_positions,
            mask,
            temperature_k=t_tensor,
        )

        loss_fit = (pred - target).pow(2).mean()
        loss = loss_fit
        if f0_batch is not None:
            loss = loss + prior_weight * F.mse_loss(mol_fractions, f0_batch)

        loss.backward()
        opt.step()

        with torch.no_grad():
            lvals = (pred - target).abs()
            j = int(lvals.argmin())
            if float(lvals[j]) < best_loss:
                best_loss = float(lvals[j])
                best_frac = mol_fractions[j].detach().clone()
                best_pred = pred[j].detach().clone()

    assert best_frac is not None and best_pred is not None
    names = [col.replace("mol_count_", "").upper() for col in mol_columns]
    recipe = {names[i]: float(best_frac[i].item()) for i in range(c)}
    return {
        "mol_fractions": best_frac.detach().cpu().tolist(),
        "mol_counts_scale": mol_count_scale,
        "predicted_conductivity": float(best_pred.item()),
        "target_conductivity": float(target_conductivity),
        "abs_error": float(abs(best_pred.item() - target_conductivity)),
        "recipe": recipe,
    }


def _fraction_key(arr: np.ndarray, decimals: int = 8) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(arr, dtype=np.float64), decimals).tolist())


def build_local_grid_candidates_simplex(
    x0: np.ndarray | list[float],
    *,
    max_pair_delta: float,
    grid_levels: int,
    min_fraction: float = 1e-8,
) -> np.ndarray:
    """
    Construct a local grid around x0 on the simplex: for each component pair (i,j),
    shift along f_i += d, f_j -= d (sum stays 1), scanning
    d in linspace(-max_pair_delta, max_pair_delta, grid_levels) and dropping points with
    any negative component. Always includes x0; deduplicates.
    """
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    if x0.ndim != 1 or x0.size == 0:
        raise ValueError("x0 must be a 1-D positive-length mol-fraction vector")
    x0 = np.clip(x0, min_fraction, 1.0)
    x0 = x0 / x0.sum()
    c = int(x0.size)
    levels = max(2, int(grid_levels))
    deltas = np.linspace(-float(max_pair_delta), float(max_pair_delta), levels)

    seen: set[tuple[float, ...]] = set()
    out_list: list[np.ndarray] = []

    def add_candidate(f: np.ndarray) -> None:
        f = np.asarray(f, dtype=np.float64).reshape(-1)
        if f.shape[0] != c:
            return
        if np.any(f < min_fraction - 1e-12):
            return
        f = np.clip(f, min_fraction, 1.0)
        s = float(f.sum())
        if s <= 0:
            return
        f = f / s
        k = _fraction_key(f)
        if k in seen:
            return
        seen.add(k)
        out_list.append(f.copy())

    add_candidate(x0)

    for i in range(c):
        for j in range(c):
            if i == j:
                continue
            for d in deltas:
                if abs(float(d)) < 1e-14:
                    continue
                f = x0.copy()
                f[i] += d
                f[j] -= d
                add_candidate(f)

    return np.stack(out_list, axis=0)


def local_grid_refine_mol_fractions(
    model: ConductivityPredictorEnhanced,
    train_config: dict,
    mol_columns: list,
    x0_fractions: np.ndarray | list[float],
    target_conductivity: float,
    device: torch.device,
    *,
    mol_count_scale: float = 1.0,
    max_seq_len: int = 1200,
    seq_len: int = 100,
    ion_pair_id: int = 0,
    temperature_k: float | None = None,
    max_pair_delta: float = 0.08,
    grid_levels: int = 7,
    eval_batch_size: int = 256,
    min_fraction: float = 1e-8,
    target_tolerance_pct: float = 2.0,
    target_tolerance_abs_floor: float = 1e-3,
) -> dict:
    """
    Using the gradient-optimized components as x0, enumerate a local grid on the simplex
    neighborhood with batched forward passes, and select the recipe minimizing
    |pred - target|.

    target_tolerance_pct is used only to label whether the result meets the relative error;
    it does not truncate the search.
    """
    c = len(mol_columns)
    x0_np = np.asarray(x0_fractions, dtype=np.float64).reshape(-1)
    if x0_np.shape[0] != c:
        raise ValueError(f"x0 length should be {c}, got {x0_np.shape[0]}")

    t_k = temperature_k if temperature_k is not None else float(train_config.get("default_temperature_k", 298.15))

    candidates = build_local_grid_candidates_simplex(
        x0_np,
        max_pair_delta=max_pair_delta,
        grid_levels=grid_levels,
        min_fraction=min_fraction,
    )
    n = int(candidates.shape[0])

    model.eval()
    pred_before, _, _, _ = model(
        torch.as_tensor(x0_np, dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(x0_np * mol_count_scale, dtype=torch.float32, device=device).unsqueeze(0),
        *build_rdf_batch(1, max_seq_len, ion_pair_id, seq_len, device),
        temperature_k=torch.full((1,), t_k, device=device, dtype=torch.float32),
    )
    pred_before_f = float(pred_before.squeeze().item())

    best_idx = 0
    best_abs = abs(pred_before_f - float(target_conductivity))
    best_pred = pred_before_f

    bs = max(1, int(eval_batch_size))
    with torch.no_grad():
        for start in range(0, n, bs):
            chunk = candidates[start : start + bs]
            b = int(chunk.shape[0])
            mol_fractions = torch.as_tensor(chunk, dtype=torch.float32, device=device)
            mol_counts = mol_fractions * float(mol_count_scale)
            ion_pair_ids, r_positions, mask = build_rdf_batch(b, max_seq_len, ion_pair_id, seq_len, device)
            t_tensor = torch.full((b,), t_k, device=device, dtype=torch.float32)
            pred, _, _, _ = model(
                mol_fractions,
                mol_counts,
                ion_pair_ids,
                r_positions,
                mask,
                temperature_k=t_tensor,
            )
            pred_np = pred.detach().float().cpu().numpy().reshape(-1)
            abs_err = np.abs(pred_np - float(target_conductivity))
            j = int(np.argmin(abs_err))
            if float(abs_err[j]) < best_abs:
                best_abs = float(abs_err[j])
                best_idx = start + j
                best_pred = float(pred_np[j])

    best_frac = candidates[best_idx].astype(np.float64)
    tgt = float(target_conductivity)
    rel_tol = max(float(target_tolerance_pct) / 100.0 * max(abs(tgt), 1e-12), float(target_tolerance_abs_floor))
    within_tol = best_abs <= rel_tol

    names = [col.replace("mol_count_", "").upper() for col in mol_columns]
    recipe = {names[i]: float(best_frac[i]) for i in range(c)}

    return {
        "mol_fractions": best_frac.astype(np.float64).tolist(),
        "mol_counts_scale": mol_count_scale,
        "predicted_conductivity": float(best_pred),
        "target_conductivity": tgt,
        "abs_error": float(best_abs),
        "recipe": recipe,
        "local_refine": {
            "num_candidates": n,
            "max_pair_delta": float(max_pair_delta),
            "grid_levels": int(grid_levels),
            "predicted_conductivity_before_local_refine": pred_before_f,
            "abs_error_before_local_refine": abs(pred_before_f - tgt),
            "target_tolerance_pct": float(target_tolerance_pct),
            "target_tolerance_effective_abs": float(rel_tol),
            "within_target_tolerance": bool(within_tol),
        },
    }


def exhaustive_grid_candidate_count(num_components: int, num_bins: int) -> int:
    """
    Number of fraction combinations on a full-simplex grid of step 1/num_bins over
    num_components components, i.e. the count of ways to write an integer m in
    [0, num_bins] as an ordered sum of num_components non-negative integers
    (stars-and-bars): C(num_bins + num_components - 1, num_components - 1).

    This is the exhaustive, discretized design space a naive screen would evaluate,
    used as the baseline for the speedup comparison.
    """
    n = int(num_bins)
    k = int(num_components) - 1
    n_choose = n + k  # choose n from (n + k)
    # C(n_choose, k) = C(n_choose, n)
    k = min(k, n_choose - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n_choose - k + i) // i
    return int(result)


def _device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_single_sample_forward_seconds(
    model: ConductivityPredictorEnhanced,
    train_config: dict,
    mol_columns: list,
    device: torch.device,
    *,
    mol_count_scale: float,
    max_seq_len: int,
    seq_len: int,
    ion_pair_id: int,
    temperature_k: float | None,
    warmup: int,
    repeats: int,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Under no_grad, measure the mean per-forward time (seconds) at batch=1, consistent with
    the inverse script, including CUDA synchronization. Returns (mean_seconds, std_seconds).
    """
    c = len(mol_columns)
    t_k = temperature_k if temperature_k is not None else float(train_config.get("default_temperature_k", 298.15))
    rng = np.random.default_rng(seed)
    x = rng.random(c).astype(np.float64)
    x = (x / x.sum()).astype(np.float32)
    mol_fractions = torch.from_numpy(x).to(device=device, dtype=torch.float32).unsqueeze(0)
    mol_counts = mol_fractions * float(mol_count_scale)
    ion_pair_ids, r_positions, mask = build_rdf_batch(1, max_seq_len, ion_pair_id, seq_len, device)
    t_tensor = torch.full((1,), t_k, device=device, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _, _, _, _ = model(
                mol_fractions,
                mol_counts,
                ion_pair_ids,
                r_positions,
                mask,
                temperature_k=t_tensor,
            )
        _device_sync(device)

        samples: list[float] = []
        for _ in range(max(1, repeats)):
            _device_sync(device)
            t0 = time.perf_counter()
            _, _, _, _ = model(
                mol_fractions,
                mol_counts,
                ion_pair_ids,
                r_positions,
                mask,
                temperature_k=t_tensor,
            )
            _device_sync(device)
            samples.append(time.perf_counter() - t0)

    arr = np.asarray(samples, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def main():
    parser = argparse.ArgumentParser(description="Conductivity inverse design: optimize mol_fractions to match target")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="trained checkpoint .pth (contains model_state_dict and config)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="RDF pretrained best_model.pt (provides the network config and initial skeleton; weights are overridden by the checkpoint)",
    )
    parser.add_argument(
        "--mol-columns",
        type=str,
        default=str(_DATA_DIR / "mol_columns.json"),
        help="path to mol_columns.json",
    )
    parser.add_argument("--target", type=float, required=True, help="target conductivity (same units as training labels, typically mS/cm)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--num-restarts", type=int, default=8, help="number of parallel random initial values; keeps the one with prediction closest to the target")
    parser.add_argument("--mol-count-scale", type=float, default=1.0, help="mol_counts = scale * mol_fractions")
    parser.add_argument("--max-seq-len", type=int, default=1200)
    parser.add_argument("--seq-len", type=int, default=100, help="effective number of RDF steps (consistent with the dataset)")
    parser.add_argument("--ion-pair-id", type=int, default=0)
    parser.add_argument("--temperature-k", type=float, default=None, help="override the default temperature; by default uses the checkpoint config's default_temperature_k")
    parser.add_argument(
        "--prior-weight",
        type=float,
        default=0.0,
        help="if --init-fractions is also given, add prior_weight * MSE(f, f0)",
    )
    parser.add_argument(
        "--init-fractions",
        type=str,
        default=None,
        help='initial mol-fraction JSON array, ordered as mol_columns, e.g. "[0.3,0.2,...]"',
    )
    parser.add_argument(
        "--output-order-csv",
        type=str,
        default=str(_DEFAULT_OUTPUT_ORDER_CSV),
        help="CSV path: the key order of mol_fractions and recipe in the output follows that file's header component columns; if the file is missing, mol_columns order is kept",
    )
    parser.add_argument(
        "--local-refine",
        action="store_true",
        help="around the gradient-optimal x0, enumerate a grid over component pairs (i,j) via f_i±Δf, f_j∓Δf, batch-predict, and pick the recipe closest to the target",
    )
    parser.add_argument(
        "--local-refine-max-pair-delta",
        type=float,
        default=0.08,
        help="max per-step transfer in the local grid (in mol-fraction scale, keeping the sum to 1 across i<->j)",
    )
    parser.add_argument(
        "--local-refine-levels",
        type=int,
        default=7,
        help="number of evenly sampled points (inclusive) for Delta f over [-max_pair_delta, max_pair_delta]",
    )
    parser.add_argument(
        "--local-refine-batch",
        type=int,
        default=256,
        help="batch size for the local-grid candidate forward passes",
    )
    parser.add_argument(
        "--target-tolerance-pct",
        type=float,
        default=2.0,
        help="relative threshold used to label whether a result meets the target %%: effective threshold is max(pct/100*|target|, 1e-3)",
    )
    parser.add_argument(
        "--benchmark-speedup",
        action="store_true",
        help="measure per-forward time at batch=1 and estimate the wall-clock cost of an exhaustive discretized grid screen over the composition simplex (see --benchmark-grid-bins); compare it against this inverse-design wall-clock time for a speedup factor",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=5,
        help="with --benchmark-speedup: number of forward warmup runs",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=32,
        help="with --benchmark-speedup: number of timed forward runs (averaged)",
    )
    parser.add_argument(
        "--benchmark-grid-bins",
        type=int,
        default=20,
        help="with --benchmark-speedup: fraction-grid resolution (step = 1/num_bins) for the hypothetical exhaustive screen over the component simplex; the screen evaluates C(num_bins + C - 1, C - 1) candidates",
    )

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mol_columns = load_mol_columns(args.mol_columns)

    init_vec = None
    if args.init_fractions:
        init_vec = np.array(json.loads(args.init_fractions), dtype=np.float32)
        if init_vec.shape[0] != len(mol_columns):
            raise SystemExit(f"--init-fractions length must be {len(mol_columns)}")

    print(f"Device: {device}")
    model, train_config = load_trained_enhanced(
        args.checkpoint, args.pretrained, mol_columns, device
    )

    benchmark_single_forward_mean_s: float | None = None
    benchmark_single_forward_std_s: float | None = None
    if args.benchmark_speedup:
        benchmark_single_forward_mean_s, benchmark_single_forward_std_s = benchmark_single_sample_forward_seconds(
            model,
            train_config,
            mol_columns,
            device,
            mol_count_scale=args.mol_count_scale,
            max_seq_len=args.max_seq_len,
            seq_len=args.seq_len,
            ion_pair_id=args.ion_pair_id,
            temperature_k=args.temperature_k,
            warmup=args.benchmark_warmup,
            repeats=args.benchmark_repeats,
        )

    t_design0 = time.perf_counter()
    out = inverse_optimize_mol_fractions(
        model,
        train_config,
        mol_columns,
        args.target,
        device,
        num_restarts=args.num_restarts,
        steps=args.steps,
        lr=args.lr,
        mol_count_scale=args.mol_count_scale,
        max_seq_len=args.max_seq_len,
        seq_len=args.seq_len,
        ion_pair_id=args.ion_pair_id,
        temperature_k=args.temperature_k,
        prior_weight=args.prior_weight,
        init_fractions=init_vec,
    )

    if args.local_refine:
        refined = local_grid_refine_mol_fractions(
            model,
            train_config,
            mol_columns,
            out["mol_fractions"],
            args.target,
            device,
            mol_count_scale=args.mol_count_scale,
            max_seq_len=args.max_seq_len,
            seq_len=args.seq_len,
            ion_pair_id=args.ion_pair_id,
            temperature_k=args.temperature_k,
            max_pair_delta=args.local_refine_max_pair_delta,
            grid_levels=args.local_refine_levels,
            eval_batch_size=args.local_refine_batch,
            target_tolerance_pct=args.target_tolerance_pct,
        )
        out.update(
            {
                "mol_fractions": refined["mol_fractions"],
                "predicted_conductivity": refined["predicted_conductivity"],
                "abs_error": refined["abs_error"],
                "recipe": refined["recipe"],
            }
        )
        out["local_refine"] = refined["local_refine"]
    t_design1 = time.perf_counter()
    inverse_design_wall_time_s = float(t_design1 - t_design0)

    if args.benchmark_speedup and benchmark_single_forward_mean_s is not None:
        mean_s = float(benchmark_single_forward_mean_s)
        num_components = len(mol_columns)
        num_candidates = exhaustive_grid_candidate_count(num_components, args.benchmark_grid_bins)
        naive_screen_wall_time_s = mean_s * float(num_candidates)
        sec_per_year = 365.25 * 24.0 * 3600.0
        naive_screen_wall_time_years = naive_screen_wall_time_s / sec_per_year
        inverse_design_wall_time_years = inverse_design_wall_time_s / sec_per_year
        if inverse_design_wall_time_s > 1e-30:
            speedup_vs_naive_screen = naive_screen_wall_time_s / inverse_design_wall_time_s
        else:
            speedup_vs_naive_screen = float("inf")
        out["speedup_benchmark"] = {
            "description": (
                "Baseline is an exhaustive, discretized screen over the component fraction "
                "simplex at step 1/num_bins: it evaluates C(num_bins + C - 1, C - 1) candidates, "
                "each with a single batch=1 forward pass of the same cost as measured here. "
                "The inverse-design wall clock is measured end-to-end (Adam optimization + "
                "optional local-grid refinement) on the same model. This is an apples-to-apples "
                "cost comparison over the same design space; it does not compare prediction "
                "quality, only search cost."
            ),
            "single_forward_batch_size": 1,
            "single_forward_mean_seconds": mean_s,
            "single_forward_std_seconds": float(benchmark_single_forward_std_s or 0.0),
            "benchmark_warmup_steps": int(args.benchmark_warmup),
            "benchmark_repeat_steps": int(args.benchmark_repeats),
            "num_components": num_components,
            "naive_screen_grid_bins": int(args.benchmark_grid_bins),
            "naive_screen_fraction_step": 1.0 / float(args.benchmark_grid_bins),
            "naive_screen_num_candidates": num_candidates,
            "naive_screen_wall_time_seconds": naive_screen_wall_time_s,
            "naive_screen_wall_time_years": naive_screen_wall_time_years,
            "inverse_design_wall_time_seconds": inverse_design_wall_time_s,
            "inverse_design_wall_time_years": inverse_design_wall_time_years,
            "speedup_vs_naive_screen_same_forward_cost": (
                speedup_vs_naive_screen
                if math.isfinite(speedup_vs_naive_screen)
                else None
            ),
            "speedup_vs_naive_screen_log10_approx": (
                math.log10(speedup_vs_naive_screen)
                if math.isfinite(speedup_vs_naive_screen) and speedup_vs_naive_screen > 0
                else None
            ),
        }

    csv_order_path = Path(args.output_order_csv)
    if csv_order_path.is_file():
        reorder_result_to_csv_columns(out, mol_columns, str(csv_order_path))
    else:
        out["component_column_order"] = [_mol_column_to_short_name(c) for c in mol_columns]
        out["output_order_source_csv"] = None

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
