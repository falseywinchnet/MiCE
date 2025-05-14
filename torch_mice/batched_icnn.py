# -*- coding: utf-8 -*-
# Copyright © 2025 Joshuah Rainstar
# License: see ../LICENSE.txt

"""
Batched Input-Convex Neural Network (ICNN) module.

Each “petal” is a small convex network; outputs are stacked over
the petal dimension. Convexity is enforced via positive weights
(softplus²) and additive convex gating.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .positive_linear import PositiveLinear3DHK
from .convex_gate     import ConvexGate

__all__ = ["BatchedICNN"]


class BatchedICNN(nn.Module):
    """
    Input-Convex Neural Network over batches of points,
    with additive convex gating to guarantee convexity.

    Args:
        in_dim:   dimensionality of each input vector D
        petals:   number of parallel convex “petals” P
        out_dim:  output dimension per petal D_out

    Input:
        x: (..., D)

    Output:
        out: (..., P, D_out)
    """
    def __init__(self, in_dim: int, petals: int, out_dim: int):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.P       = petals

        D    = in_dim
        D_out = out_dim
        self.d1 = 2 * D
        self.d2 = D_out

        # core convex layers
        self.layer0   = PositiveLinear3DHK(petals, D,      self.d1)
        self.layer1   = PositiveLinear3DHK(petals, self.d1, self.d2)
        self.res_proj = PositiveLinear3DHK(petals, 2 * D,  self.d2)

        # convex gates
        self.gate0_net      = ConvexGate(D, self.d1)
        self.gate1_net      = ConvexGate(D, self.d2)
        self.extra_gate0_nets = nn.ModuleList(
            [ConvexGate(D, self.d1) for _ in range(self.P)]
        )
        self.extra_gate1_nets = nn.ModuleList(
            [ConvexGate(D, self.d2) for _ in range(self.P)]
        )

        self.out_bias = nn.Parameter(torch.zeros(self.P, self.d2))
        self.act      = nn.Softplus()

    def forward(self, x_p: torch.Tensor, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_p:    (P, N, D_in)  — per-petal warped inputs
            x_flat: (N, D_in)     — original inputs for gates
        Returns:
            (N, P, D_out)
        """
        P, N, D = x_p.shape
        assert P == self.P and D == self.in_dim
        assert x_flat.shape == (N, D)

        # 1) main gates
        g0 = self.gate0_net(x_flat).unsqueeze(0).expand(P, N, self.d1)
        g1 = self.gate1_net(x_flat).unsqueeze(0).expand(P, N, self.d2)

        # 2) extra gates
        extra0 = torch.stack([g(x_flat) for g in self.extra_gate0_nets], dim=0)
        extra1 = torch.stack([g(x_flat) for g in self.extra_gate1_nets], dim=0)

        # 3) layer0 + gates
        z0 = self.act(self.layer0(x_p) + g0)
        z0 = self.act(z0 + extra0)

        # 4) layer1 + gates
        z1 = self.act(self.layer1(z0) + g1)
        z1 = self.act(z1 + extra1)

        # 5) residual
        res_in = torch.cat([x_p, x_p], dim=-1)      # (P, N, 2*D)
        res    = self.res_proj(res_in)              # (P, N, d2)

        # 6) combine + bias
        out_p = self.act(z1 + res) + self.out_bias.unsqueeze(1)  # (P, N, d2)

        # 7) permute to (N, P, D_out)
        return out_p.permute(1, 0, 2)
