# -*- coding: utf-8 -*-
# Copyright © 2025 Joshuah Rainstar
# License: see ../LICENSE.txt


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .atlas_projector import SingleStiefelProjector
from .affine_norm import BatchAffineNorm

class GeometricConvexEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 512, expand_factor: int = 4):
        super().__init__()
        self.in_dim = embed_dim
        self.expanded_dim = embed_dim * expand_factor

        # Learn token representations in higher-dimensional space
        self.embed_table = nn.Embedding(vocab_size, self.expanded_dim)

        # Project down to embed_dim via SO(embed_dim)
        self.projector = SmoothStiefelProjector(self.in_dim)

        # Contract via learnable linear map or stability layer
        self.contractor = BatchAffineNorm(self.in_dim)

        # Linear contraction: optional alternative to FrozenAffine
        # self.contractor = nn.Linear(self.in_dim, self.in_dim)

        # Expansion matrix: learnable expansion to match projection input
        self.expander = nn.Linear(self.expanded_dim, self.in_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, S) token indices
        Returns: (B, S, embed_dim)
        """
        raw_embed = self.embed_table(idx)         # (B, S, 2D)
        expanded = self.expander(raw_embed)       # (B, S, D)
        projected = self.projector(expanded)      # (B, S, D)
        contracted = self.contractor(projected)   # (B, S, D)
        return contracted
