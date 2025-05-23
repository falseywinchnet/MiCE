# -*- coding: utf-8 -*-
# Copyright © 2025 Joshuah Rainstar
# License: see ../LICENSE.txt


import torch
import torch.nn as nn
import torch.nn.functional as F


from .batched_icnn import BatchedICNN
from .affine_norm import BatchAffineNorm

__all__ = ["ConvexFeatureProjector"]


class ConvexFeatureProjector(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, icnn_dim: int = None):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.icnn_dim = icnn_dim or self.head_dim

        # Swap LayerNorm with convex-respecting BatchAffineNorm
        self.norm = BatchAffineNorm(embed_dim)

        # Convex petal extractor
        self.convex_extractor = BatchedICNN(
            in_dim=embed_dim,
            petals=num_heads,
            out_dim=self.icnn_dim
        )

        self._last_head_outputs = None

    def forward(self, x):
        B, T, D = x.shape
        x_norm = self.norm(x)  # convex-safe norm

        x_flat = x_norm.reshape(B * T, D)
        x_p = x_flat.unsqueeze(0).expand(self.num_heads, -1, -1)

        icnn_out = self.convex_extractor(x_p, x_flat)  # (N, P, D_out)
        icnn_out = icnn_out.reshape(B, T, -1)  # (B, T, num_heads * D_out)

        # Cache head-wise outputs for orthogonal loss if needed
        self._last_head_outputs = torch.unbind(
            icnn_out.view(B, T, self.num_heads, self.icnn_dim), dim=2
        )
        return icnn_out

    def orthogonal_loss(self, lambda_reg=1.0):
        if self._last_head_outputs is None:
            return 0.0
        head_outputs = self._last_head_outputs
        loss = 0.0
        B, T, D = head_outputs[0].shape
        for i in range(len(head_outputs)):
            Zi = head_outputs[i].reshape(B * T, D)
            for j in range(i + 1, len(head_outputs)):
                Zj = head_outputs[j].reshape(B * T, D)
                gram = torch.matmul(Zi.T, Zj)
                loss += torch.norm(gram, p='fro') ** 2
        return lambda_reg * loss
