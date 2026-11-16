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
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def ensemble_metrics_with_loo_stderr(y_true: np.ndarray, preds_per_model: np.ndarray) -> dict[str, float]:

    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(preds_per_model, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != y.size:
        raise ValueError(f"preds_per_model must be (K, N) with N={y.size}, got {p.shape}")
    k = int(p.shape[0])
    mean_pred = p.mean(axis=0)
    out: dict[str, float] = {
        "mae": float(mean_absolute_error(y, mean_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, mean_pred))),
        "r2": float(r2_score(y, mean_pred)),
    }
    if k <= 1:
        out.update({"mae_stderr": 0.0, "rmse_stderr": 0.0, "r2_stderr": 0.0})
        return out
    mae_list: list[float] = []
    rmse_list: list[float] = []
    r2_list: list[float] = []
    for j in range(k):
        idx = [i for i in range(k) if i != j]
        sub_mean = p[idx].mean(axis=0)
        mae_list.append(float(mean_absolute_error(y, sub_mean)))
        rmse_list.append(float(np.sqrt(mean_squared_error(y, sub_mean))))
        r2_list.append(float(r2_score(y, sub_mean)))
    out["mae_stderr"] = float(np.std(mae_list, ddof=1))
    out["rmse_stderr"] = float(np.std(rmse_list, ddof=1))
    out["r2_stderr"] = float(np.std(r2_list, ddof=1))
    return out
