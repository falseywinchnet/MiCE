# -*- coding: utf-8 -*-
# Copyright © 2025 Joshuah Rainstar
# License: see ../LICENSE.txt


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .atlas_projector import SingleStiefelProjector
from .affine_norm import BatchAffineNorm
from .positive_linear import PositiveLinearHK

__all__ = ["PositiveEmbeddingHK","GeometricConvexEmbedding"]



class PositiveEmbeddingHK(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.raw = nn.Parameter(torch.empty(vocab_size, embed_dim))
        with torch.no_grad():
            mean = math.log(math.sqrt(2.0 / vocab_size))
            nn.init.normal_(self.raw, mean=mean, std=0.2)

    @property
    def weight(self):
        return F.softplus(self.raw)  # ensure positivity

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, S) — long tensor of token indices
        return: (B, S, D) — positive embedding vectors
        """
        return self.weight[idx]

class GeometricConvexEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, expand_factor: int = 4):
        super().__init__()
        self.D = embed_dim
        self.E = embed_dim * expand_factor

        self.embed = PositiveEmbeddingHK(vocab_size, self.E)         # V × E
        self.reduce = PositiveLinearHK(self.E, self.D, bias=False)  # E → D
        self.projector = SingleStiefelProjector(self.D)               # SO(E)
        self.expand = PositiveLinearHK(self.D, self.E, bias=False)  # E → D
        self.reduce2 = PositiveLinearHK(self.E, self.D, bias=False)  # E → D
        self.norm = BatchAffineNorm(self.D)
        
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, S) long
        return: (B, S, D) — strictly convex, unit-norm embeddings
        """
        x = self.embed(idx)                    #(B, S, D), positive
        x = self.reduce(x)                 # (B, S, E), positive
        x = self.projector(x)                # (B, S, E), rotated
        x = self.expand(x)                 # (B, S, D), positive
        x = self.reduce2(x)
        x = self.norm(x)
        return x


