# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Shared architecture blocks."""

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from torch_utils.ops import bias_act


class ResidualBlock(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x) + x) / np.sqrt(2)


class FullyConnectedLayer(nn.Module):
    def __init__(
        self,
        in_features: int,              # Number of input features.
        out_features: int,             # Number of output features.
        bias: bool  = True,            # Apply additive bias before the activation function?
        activation: str   = 'linear',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier: float = 1.0,    # Learning rate multiplier.
        weight_init: float = 1.0,      # Initial standard deviation of the weight tensor.
        bias_init: float = 0.0,        # Initial value for the additive bias.
    ):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self) -> str:
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


class MLP(nn.Module):
    def __init__(
        self,
        features_list: list[int],    # Number of features in each layer of the MLP.
        activation: str = 'linear',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier: float = 1.0,  # Learning rate multiplier.
        linear_out: bool = False     # Use the 'linear' activation function for the output layer?
    ):
        super().__init__()
        num_layers = len(features_list) - 1
        self.num_layers = num_layers
        self.out_dim = features_list[-1]

        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            if linear_out and idx == num_layers-1:
                activation = 'linear'
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' if x is sequence of tokens, shift tokens to batch and apply MLP to all'''
        shift2batch = (x.ndim == 3)

        if shift2batch:
            B, K, C = x.shape
            x = x.flatten(0,1)

        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        if shift2batch:
            x = x.reshape(B, K, -1)

        return x
