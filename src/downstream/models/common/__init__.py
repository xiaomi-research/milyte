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
"""Shared common base: datasets/splitting, loss functions, component
physicochemical features, model architecture blocks, and ensembling metrics.

Final property predictors import shared implementations from this package rather than
duplicating historical code.
"""
from .data import ConductivityDataset, split_data_by_conductivity  # noqa: F401
from .losses import weighted_mse_loss, relative_error_loss, hybrid_loss  # noqa: F401
from .components import (  # noqa: F401
    COMPONENT_PROPERTIES,
    PROPERTY_NAMES,
    extract_physicochemical_features,
)
from .arch import (  # noqa: F401
    ArrheniusTemperatureEmbedding,
    EnhancedCompositionFeatureExtractor,
    FeatureModulationFusion,
    RegressionHead,
)
from .metrics import ensemble_metrics_with_loo_stderr  # noqa: F401
