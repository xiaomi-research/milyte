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
import torch.nn.functional as F

def weighted_mse_loss(pred, target, weight_type='dynamic'):

    if weight_type == 'dynamic':

        weights = (target / target.mean()) ** 2
    elif weight_type == 'linear':

        weights = target / target.mean()
    elif weight_type == 'threshold':

        threshold = torch.quantile(target, 0.75)
        weights = torch.where(target >= threshold, 
                            torch.tensor(3.0, device=target.device), 
                            torch.tensor(1.0, device=target.device))
    else:
        weights = torch.ones_like(target)

    weights = weights / weights.mean()

    squared_error = (pred - target) ** 2
    weighted_loss = (squared_error * weights).mean()
    
    return weighted_loss

def relative_error_loss(pred, target, epsilon=1e-6):

    relative_error = torch.abs((pred - target) / (target + epsilon))
    return relative_error.mean()

def hybrid_loss(pred, target, alpha=0.5, beta=0.5, weight_type='dynamic'):

    mse_loss = weighted_mse_loss(pred, target, weight_type=weight_type)
    rel_loss = relative_error_loss(pred, target)
    
    return alpha * mse_loss + beta * rel_loss
