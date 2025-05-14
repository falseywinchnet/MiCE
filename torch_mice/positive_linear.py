# -*- coding: utf-8 -*-
# Copyright © 2025 Joshuah Rainstar
# License: see ../LICENSE.txt

"""
Positive linear layers with Hoedt–Klambauer parameterization,
ensuring strictly positive weights via softplus².
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PositiveLinearHK",
    "PositiveLinear3DHK",
]

class PositiveLinearHK(nn.Module): #Hoedt–Klambauer MUST be used always
    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        self.raw = nn.Parameter(torch.empty(d_out, d_in))
        if bias:
            self.bias = nn.Parameter(torch.zeros(d_out))
        else:
            self.register_parameter('bias', None)
        with torch.no_grad():
            mean = math.log(math.sqrt(2.0 / d_in))
            nn.init.normal_(self.raw, mean=mean, std=0.2)

    @property
    def weight(self):
        return F.softplus(self.raw)  # ensures strict positivity

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class PositiveLinear3DHK(nn.Module):
    """
    TorchScript-safe version of a positive linear map over petals (P, N, D_in → D_out),
    using softplus² and optional Frobenius normalization *during training only*.
    """
    def __init__(self, petals: int, D_in: int, D_out: int, norm_target: float = None):
        super().__init__()
        self.P = petals
        self.D_in = D_in
        self.D_out = D_out
        self.norm_target = norm_target

        self.raw = nn.Parameter(torch.empty(petals, D_out, D_in))
        self.bias = nn.Parameter(torch.zeros(petals, D_out))

        with torch.no_grad():
            mean = math.log(math.sqrt(2.0 / D_in))
            nn.init.normal_(self.raw, mean=mean, std=0.2)

    def compute_weight(self) -> torch.Tensor:
        W = F.softplus(self.raw)
        if self.norm_target is not None and self.training:
            W_squared = W ** 2
            norms = torch.sqrt((W_squared).sum(dim=(1, 2), keepdim=True) + 1e-12)  # (P,1,1)
            W = W * (self.norm_target / norms)
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (P, N, D_in)
        W = self.compute_weight()                    # (P, D_out, D_in)
        out = torch.bmm(x, W.transpose(1, 2))        # (P, N, D_out)
        out = out + self.bias.unsqueeze(1)           # (P, N, D_out)
        return out
