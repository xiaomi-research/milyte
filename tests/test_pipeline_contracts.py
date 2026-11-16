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
"""CPU-only contract tests for data/config/CLI wiring.

These tests do not require the Hugging Face assets.  Checks that need the
downloaded datasets or weights are skipped when those files are absent, so the
public repository can be tested immediately after cloning.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = REPO_ROOT / "src" / "downstream" / "models"
PRETRAIN = REPO_ROOT / "src" / "pretrain"
DATA = REPO_ROOT / "data" / "downstream"
PRETRAIN_DATA = REPO_ROOT / "data" / "pretrain"
CKPT_DIR = REPO_ROOT / "data" / "checkpoints"

if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

_HF_HINT = "Download the assets from https://huggingface.co/xiaomi-research/milyte into data/."


def _load_mol_columns():
    with open(MODELS / "mol_columns.json", encoding="utf-8") as f:
        raw = json.load(f)
    return list(raw["mol_columns"] if isinstance(raw, dict) else raw)


def _run_help(script_path: Path):
    result = subprocess.run(
        [sys.executable, str(script_path), "-h"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_entry_points_have_help():
    for script in (
        MODELS / "rdf_backbone.py",
        MODELS / "recipe_inference.py",
        MODELS / "inverse_design.py",
        MODELS / "train_dcr_dual_temperature.py",
    ):
        _run_help(script)


def test_pretrain_config_has_expected_architecture():
    from rdf_backbone import _flatten_config

    with open(PRETRAIN / "train_config.yaml", encoding="utf-8") as f:
        cfg = _flatten_config(yaml.safe_load(f))

    assert cfg["num_layers"] == 5
    assert cfg["dim_feedforward"] == 768
    assert cfg["dropout"] == pytest.approx(0.15)
    assert cfg["d_model"] == 256
    assert cfg["max_seq_len"] == 1200
    assert cfg["batch_size"] == 24
    assert cfg["learning_rate"] == pytest.approx(7e-5)
    assert cfg["num_epochs"] == 80
    assert cfg["early_stopping_patience"] == 12
    assert cfg["use_se3_equivariance"] is True


def test_pretrain_config_path_resolution():
    from rdf_backbone import load_train_config, _resolve_pretrain_paths

    cfg = _resolve_pretrain_paths(load_train_config(), data_dir=None, save_dir=None)
    for key in ("train_csv", "val_csv", "test_csv"):
        assert Path(cfg[key]).parent == PRETRAIN_DATA, cfg[key]
    assert Path(cfg["save_dir"]) == CKPT_DIR


@pytest.mark.skipif(
    not (CKPT_DIR / "config.json").is_file(),
    reason="data/checkpoints/config.json not found. " + _HF_HINT,
)
def test_pretrain_config_matches_released_checkpoint():
    from rdf_backbone import _flatten_config

    with open(PRETRAIN / "train_config.yaml", encoding="utf-8") as f:
        yaml_cfg = _flatten_config(yaml.safe_load(f))
    with open(CKPT_DIR / "config.json", encoding="utf-8") as f:
        ckpt_cfg = json.load(f)

    keys = (
        "num_components",
        "num_ion_pairs",
        "component_hidden_dim",
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "dropout",
        "max_seq_len",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "num_epochs",
        "warmup_epochs",
        "early_stopping_patience",
        "min_delta",
        "save_interval",
    )
    for key in keys:
        assert yaml_cfg[key] == pytest.approx(ckpt_cfg[key]), (
            f"train_config.yaml[{key!r}]={yaml_cfg[key]!r} disagrees with "
            f"data/checkpoints/config.json[{key!r}]={ckpt_cfg[key]!r}"
        )


@pytest.mark.skipif(
    not (CKPT_DIR / "best_model.pt").is_file(),
    reason="data/checkpoints/best_model.pt not found. " + _HF_HINT,
)
def test_released_checkpoint_matches_code_architecture():
    """The released backbone must be the v3/SE(3) variant, not legacy v2_macro."""
    ckpt = torch.load(CKPT_DIR / "best_model.pt", map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    assert "config" in ckpt
    assert any(k.startswith("component_encoder.encoder.") for k in state_dict), (
        "best_model.pt uses the legacy mol_encoder/macro_encoder component encoder; "
        "it cannot be loaded by the current RDFPretrainModel."
    )
    assert not any("mol_encoder" in k for k in state_dict)
    assert ckpt["config"].get("use_se3_equivariance") is True


@pytest.mark.skipif(
    not (DATA / "conductivity.csv").is_file() or not (DATA / "free_energy.csv").is_file(),
    reason="data/downstream CSVs not found. " + _HF_HINT,
)
def test_recipe_inference_dataframe_contract():
    import recipe_inference as ri

    mol_columns = _load_mol_columns()
    df = pd.read_csv(DATA / "conductivity.csv").head(3).fillna(0.0)

    cond_samples = ri.dataframe_to_conductivity_samples(df, mol_columns)
    assert len(cond_samples) == len(df)
    for sample in cond_samples:
        assert set(sample["atom_counts"]) == {
            c.replace("mol_count_", "").lower() for c in mol_columns
        }
        assert "conductivity" in sample and "ratio_str" in sample

    fe_df = pd.read_csv(DATA / "free_energy.csv").head(3).fillna(0.0)
    fe_samples = ri.dataframe_to_free_energy_samples(fe_df, mol_columns)
    assert len(fe_samples) == len(fe_df)
    for sample in fe_samples:
        assert len(sample["coordination_n"]) == len(mol_columns)
        assert "free_energy" in sample


@pytest.mark.skipif(
    not all((DATA / name).is_file() for name in
            ("conductivity.csv", "viscosity.csv", "dcr.csv", "free_energy.csv")),
    reason="data/downstream CSVs not found. " + _HF_HINT,
)
def test_mol_columns_match_feature_columns():
    mol_columns = _load_mol_columns()
    names = [c.replace("mol_count_", "") for c in mol_columns]

    for csv_name in ("conductivity.csv", "viscosity.csv", "dcr.csv", "free_energy.csv"):
        header = pd.read_csv(DATA / csv_name, nrows=0).columns.str.lstrip("﻿")
        missing = [name for name in names if name not in header]
        assert not missing, f"{csv_name} is missing component columns: {missing}"

    from free_energy_predictor import mol_column_to_coord_column

    fe_header = pd.read_csv(DATA / "free_energy.csv", nrows=0).columns
    missing_coord = [
        mol_column_to_coord_column(c)
        for c in mol_columns
        if mol_column_to_coord_column(c) not in fe_header
    ]
    assert not missing_coord, f"free_energy.csv missing coordination columns: {missing_coord}"


def test_data_and_weights_are_not_tracked():
    """Datasets and weights live on Hugging Face, never in Git."""
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a Git checkout")
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = result.stdout.splitlines()
    forbidden = [
        p for p in tracked
        if p.startswith("data/") or p.endswith((".pt", ".pth", ".npy"))
    ]
    assert not forbidden, f"data/weights must not be tracked: {forbidden[:10]}"
