# -*- coding: utf-8 -*-
# Copyright © 2025 Joshuah Rainstar
# License: see ../LICENSE.txt

"""
Convex gate:  
g(x) = 1 − exp(−softplus(Wx + b)) ∈ (0,1)^out_dim  
Ensures a convex, bounded gating signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positive_linear import PositiveLinearHK


__all__ = ["ConvexGate"]

class ConvexGate(nn.Module):
    """
    Convex & bounded gate:
        g(x) = 1 - exp(-softplus(Wx + b)) ∈ (0,1)^out_dim
    """
    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.lin = PositiveLinearHK(in_dim, out_dim, bias=True)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim)
        u = self.softplus(self.lin(x))       # (..., out_dim), convex ≥ 0
        return 1.0 - torch.exp(-u)           # (..., out_dim), convex ∈ (0,1)
