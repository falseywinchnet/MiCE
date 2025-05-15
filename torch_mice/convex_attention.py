# -*- coding: utf-8 -*-
# Copyright © 2025 Joshuah Rainstar
# License: see ../LICENSE.txt

import torch
import torch.nn as nn
import torch.nn.functional as F


from .positive_linear import PositiveLinearHK
from .atlas_projector import SingleStiefelProjector
from .affine_norm import BatchAffineNorm

__all__ = ["ConvexExpansionAttention","ConvexContractionAttention"]

class ConvexExpansionAttention(nn.Module):
    def __init__(self, D: int, gamma: float = 5.0):
        super().__init__()
        assert D % 3 == 0, "Input D must be divisible by 3"
        self.D = D
        self.d = D // 3
        self.gamma = gamma

        self.q_proj = nn.Sequential(
            PositiveLinearHK(self.d, 3 * self.d),
            SingleStiefelProjector(3 * self.d)
        )
        self.k_proj = nn.Sequential(
            PositiveLinearHK(self.d, 3 * self.d),
            SingleStiefelProjector(3 * self.d)
        )
        self.v_proj = nn.Sequential(
            PositiveLinearHK(self.d, 3 * self.d),
            SingleStiefelProjector(3 * self.d)
        )

        # post-attention projection: 3d → 9d
        self.post = nn.Sequential(
            PositiveLinearHK(3 * self.d, 9 * self.d),
            BatchAffineNorm(9 * self.d)
        )

    def forward(self, x):  # x: (B, T, D) = (B, T, 3d)
        B, T, D = x.shape
        d = self.d
        assert D == 3 * d

        Q_base = x[:, :, 0*d : 1*d]
        K_base = x[:, :, 1*d : 2*d]
        V_base = x[:, :, 2*d : 3*d]

        Q = self.q_proj(Q_base)  # (B, T, 3d)
        K = self.k_proj(K_base)
        V = self.v_proj(V_base)

        dots    = torch.einsum("b t i, b t j -> b t i j", Q, K)         # (B, T, 3d, 3d)
        scores  = torch.sigmoid(self.gamma * dots)
        weights = scores / (scores.sum(dim=-1, keepdim=True) + 1e-8)

        attended = torch.einsum("b t i j, b t j -> b t i", weights, V)  # (B, T, 3d)

        return self.post(attended)  # (B, T, 9d)



class ConvexContractionAttention(nn.Module):
    """
    (B, T, 3*D) → (B, T, D)
    - Let d = D = (input_dim // 3)
    - For each i in [0..d-1]:
        Qi = x[..., i]            # (B,T)
        Ki = x[..., d + i]
        Vi = x[..., 2*d + i]
      → project each via PositiveLinearHK(1→3) + SingleStiefelProjector(3)
      → compute kernel scores over those 3 “mini‐heads”
      → collapse to one scalar per token
    - Stack all d scalars → (B, T, d)
    - Final BatchAffineNorm for stability
    """
    def __init__(self, D_in: int, gamma: float = 5.0):
        super().__init__()
        assert D_in % 3 == 0, "D_in must be divisible by 3"
        self.d = D_in // 3
        self.gamma = gamma

        # per-output‐channel small Q/K/V projectors + rotations
        self.q_blocks = nn.ModuleList([
            nn.Sequential(PositiveLinearHK(1, 3), SingleStiefelProjector(3), BatchAffineNorm(3))
            for _ in range(self.d)
        ])
        self.k_blocks = nn.ModuleList([
            nn.Sequential(PositiveLinearHK(1, 3), SingleStiefelProjector(3), BatchAffineNorm(3))
            for _ in range(self.d)
        ])
        self.v_blocks = nn.ModuleList([
            nn.Sequential(PositiveLinearHK(1, 3), SingleStiefelProjector(3), BatchAffineNorm(3))
            for _ in range(self.d)
        ])

        # final normalization
        self.norm = BatchAffineNorm(self.d)

    def forward(self, x):
        # x: (B, T, 3*d)
        B, T, C = x.shape
        d = self.d
        assert C == 3 * d

        outs = []
        for i in range(d):
            # pick the i-th element from each third
            Qi = x[...,         i    ].unsqueeze(-1)  # (B,T,1)
            Ki = x[...,     d + i     ].unsqueeze(-1)
            Vi = x[..., 2*d + i     ].unsqueeze(-1)

            # project + rotate + norm → each (B,T,3)
            Qp = self.q_blocks[i](Qi)
            Kp = self.k_blocks[i](Ki)
            Vp = self.v_blocks[i](Vi)

            # kernelized 3‐head attention per token
            scores  = torch.sigmoid(self.gamma * (Qp * Kp))                   # (B,T,3)
            weights = scores / (scores.sum(dim=-1, keepdim=True) + 1e-8)
            yi      = (weights * Vp).sum(dim=-1, keepdim=True)                # (B,T,1)

            outs.append(yi)

        out = torch.cat(outs, dim=-1)  # (B, T, d)
        return self.norm(out)
