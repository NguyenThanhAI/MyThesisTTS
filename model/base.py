# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import numpy as np
import torch
import torch.nn as nn


class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        """
        Returns number of trainable parameters of the module.
        """
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params


    def relocate_input(self, x: list):
        """
        Relocates provided tensors to the same device set for the module.
        """
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x
    

class LayerNorm(BaseModule):
    def __init__(self, channels, eps=1e-4):
        super(LayerNorm, self).__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(data=torch.ones(channels))
        self.beta = torch.nn.Parameter(data=torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(input=x, dim=1, keepdim=True)
        variance = torch.mean(input=(x - mean)**2, dim=1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
    

class SinusoidalPositionalEncoding(BaseModule):
    """
    PE(pos, 2i) = sin(pos / 10000^(2i / dim))
    PE(pos, 2i + 1) = cos(pos / 10000^(2i / dim))
    """

    def __init__(self, dim: int, dropout: float=0.1, max_len: int=5000):
        assert dim % 2 == 0, print("[ERROR]: dim: {} must be an even number".format(dim))
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        half_dim = dim // 2
        div_term = torch.exp(2 * torch.arange(0, half_dim, dtype=torch.float) * (-math.log(10000.0) / dim)).unsqueeze(0)
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_length = x.size(1)
        x = x + self.pe[:seq_length, :].unsqueeze(0)
        return self.dropout(x)
    

class ConvNorm(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor):
        x = x.contiguous()
        x = self.conv(x * x_mask)


class AffineLinear(BaseModule):
    def __init__(self, in_dim, out_dim):
        super(AffineLinear, self).__init__()
        affine = nn.Linear(in_dim, out_dim)
        self.affine = affine

    def forward(self, input):
        return self.affine(input)


class StyleAdaptiveLayerNorm(BaseModule):
    def __init__(self, in_channel, style_dim):
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.in_channel = in_channel
        self.norm = LayerNorm(channels=in_channel)

        self.style = AffineLinear(style_dim, in_channel * 2)
        self.style.affine.bias.data[:in_channel] = 1
        self.style.affine.bias.data[in_channel:] = 0

    def forward(self, input, style_code):
        # style
        style = self.style(style_code).unsqueeze(1)
        gamma, beta = style.chunk(2, dim=-1)
        gamma = gamma.transpose(1, 2)
        beta = beta.transpose(1, 2)

        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta  =       0.1 * torch.tanh(beta)
        
        out = self.norm(input)
        out = gamma * out + beta
        return out

    

if __name__ == "__main__":
    module = BaseModule()
    print(f"module: {module}")