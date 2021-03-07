# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for offline RL."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import datetime
import re

import numpy as np


import torch
import torch

def clip_by_eps(x, spec, eps=0.0):
  return torch.clamp(
      x, min=spec.minimum + eps, max=spec.maximum - eps)

# TODO: add customized gradient
def clip_v2(x, low, high):
  """Clipping with modified gradient behavior."""
  value = torch.min(torch.max(x, low * torch.ones_like((x))), high * torch.ones_like(x))
#   def grad(dy):
#     if_y_pos = torch.gt(dy, 0.0).type(torch.float32)
#     if_x_g_low = torch.gt(x, low).type(torch.float32)
#     if_x_l_high = torch.le(x, high).type(torch.float32)
#     return (if_y_pos * if_x_g_low +
#             (1.0 - if_y_pos) * if_x_l_high) * dy
#   return value, grad
  return value


# class clip_v2(torch.autograd.Function):
#   @staticmethod
#   def forward(ctx, x):
#     ctx.save_for_backward(x)
#     return torch.min(torch.max(x, 0. * torch.ones_like((x))), 500. * torch.ones_like(x))
#   @staticmethod
#   def backward(ctx, grad_output):
#     x, = ctx.saved_tensors
#     grad_cpy = grad_output.clone()
#     if_y_pos = torch.gt(grad_cpy, 0.0).type(torch.float32)
#     if_x_g_low = torch.gt(x, 0.).type(torch.float32)
#     if_x_l_high = torch.le(x, 500.).type(torch.float32)
#     return (if_y_pos * if_x_g_low +
#             (1.0 - if_y_pos) * if_x_l_high) * grad_cpy

def soft_relu(x):
  """Compute log(1 + exp(x))."""
  # Note: log(sigmoid(x)) = x - soft_relu(x) = - soft_relu(-x).
  #       log(1 - sigmoid(x)) = - soft_relu(x)
  return torch.log(1.0 + torch.exp(-torch.abs(x))) + torch.max(x, torch.zeros_like(x))

