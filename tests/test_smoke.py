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
"""Repository layout / artifact smoke tests.

The public repository contains code only.  Datasets and model weights are hosted
on Hugging Face and are expected under ``data/`` after download (see README).
Checks that need those assets are skipped with an actionable message when the
assets are absent, so a fresh clone runs green.
"""

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN = REPO_ROOT / "src" / "pretrain"
DOWNSTREAM = REPO_ROOT / "src" / "downstream"
MODELS = DOWNSTREAM / "models"

REPO_DATA = REPO_ROOT / "data"
PRETRAIN_DATA = REPO_DATA / "pretrain"
DOWNSTREAM_DATA = REPO_DATA / "downstream"
CKPT_DIR = REPO_DATA / "checkpoints"
OUTPUTS = MODELS / "outputs"

_HF_HINT = (
    "Assets are distributed via Hugging Face "
    "(https://huggingface.co/xiaomi-research/milyte); download them into data/."
)


def _missing(*paths: Path) -> bool:
    return any(not p.exists() for p in paths)


requires_pretrain_data = pytest.mark.skipif(
    _missing(PRETRAIN_DATA / "pretrain_rdf_train.csv",
             PRETRAIN_DATA / "pretrain_rdf_val.csv",
             PRETRAIN_DATA / "pretrain_rdf_test.csv"),
    reason="Pretraining CSVs not found under data/pretrain/. " + _HF_HINT,
)

requires_pretrain_ckpt = pytest.mark.skipif(
    not (CKPT_DIR / "best_model.pt").is_file(),
    reason="Pretrained backbone not found under data/checkpoints/. " + _HF_HINT,
)

requires_downstream_data = pytest.mark.skipif(
    _missing(DOWNSTREAM_DATA / "conductivity.csv",
             DOWNSTREAM_DATA / "viscosity.csv",
             DOWNSTREAM_DATA / "free_energy.csv",
             DOWNSTREAM_DATA / "dcr.csv"),
    reason="Downstream CSVs not found under data/downstream/. " + _HF_HINT,
)

requires_downstream_weights = pytest.mark.skipif(
    not OUTPUTS.is_dir(),
    reason=(
        "Downstream fold checkpoints are not in the repository; download them from "
        "Hugging Face or train the models first (see README 'Stage II')."
    ),
)


def test_root_layout():
    assert (REPO_ROOT / "README.md").is_file()
    assert (REPO_ROOT / "LICENSE").is_file()
    assert (REPO_ROOT / "environment.yml").is_file()
    assert (REPO_ROOT / "requirements.txt").is_file()
    assert PRETRAIN.is_dir()
    assert DOWNSTREAM.is_dir()


def test_pretrain_config_present():
    assert (PRETRAIN / "train_config.yaml").is_file()


@requires_pretrain_ckpt
def test_pretrain_checkpoint_present():
    assert (CKPT_DIR / "best_model.pt").is_file()
    assert (CKPT_DIR / "config.json").is_file()
    assert (CKPT_DIR / "mol_columns.json").is_file()
    assert (CKPT_DIR / "ion_pair_vocab.json").is_file()


@requires_pretrain_data
def test_pretrain_data_present():
    assert (PRETRAIN_DATA / "pretrain_rdf_train.csv").is_file()
    assert (PRETRAIN_DATA / "pretrain_rdf_val.csv").is_file()
    assert (PRETRAIN_DATA / "pretrain_rdf_test.csv").is_file()


@requires_downstream_data
def test_downstream_data_present():
    for name in ("conductivity.csv", "dcr.csv", "free_energy.csv", "viscosity.csv"):
        assert (DOWNSTREAM_DATA / name).is_file(), name
    assert (DOWNSTREAM_DATA / "mol_columns.json").is_file()
    assert (DOWNSTREAM_DATA / "ion_pair_vocab.json").is_file()


@requires_downstream_weights
def test_downstream_conductivity_weights_present():
    for i in range(1, 6):
        assert (OUTPUTS / f"conductivity_enhanced_fold{i}.pth").is_file(), f"fold{i}"


@requires_downstream_weights
def test_downstream_property_weights_present():
    for name in ("free_energy", "viscosity"):
        for i in range(1, 6):
            assert (OUTPUTS / f"{name}_enhanced_fold{i}.pth").is_file(), f"{name} fold{i}"

    for i in range(1, 6):
        assert (OUTPUTS / f"dcr_dual_temperature_fold{i}.pth").is_file(), f"dcr fold{i}"


def test_modules_import():
    """Every released entry point must be importable from the models dir."""
    models_dir = str(MODELS)
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    for module in (
        "common",
        "rdf_backbone",
        "conductivity_predictor",
        "viscosity_predictor",
        "free_energy_predictor",
        "dcr_dual_temperature_predictor",
        "recipe_inference",
        "inverse_design",
    ):
        importlib.import_module(module)


def test_path_resolution_math():
    """Data and checkpoints resolve to <repo>/data, independent of the CWD."""
    repo_from_models = MODELS.resolve().parents[2]
    assert repo_from_models / "data" / "checkpoints" == CKPT_DIR
    assert repo_from_models / "data" / "pretrain" == PRETRAIN_DATA
    assert repo_from_models / "data" / "downstream" == DOWNSTREAM_DATA
