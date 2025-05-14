# -*- coding: utf-8 -*-
# Copyright © 2025 Joshuah Rainstar
# License: see ../LICENSE.txt

"""
VectorHull: Convex vector-valued function via overlapping shifted max-of-means fusion.
Each input is passed through a BatchedICNN “petal” ensemble, then grouped into
circular overlapping pairs, averaged, shifted by a learnable bias, and max-combined
to guarantee convexity without exponentials.
"""

import torch
import torch.nn as nn

from .batched_icnn import BatchedICNN

__all__ = ["VectorHull"]


class VectorHull(nn.Module):
    """
    Args:
        in_dim:   input feature dimensionality D
        petals:   number of parallel convex “petals” P
        out_dim:  output dimensionality per petal (defaults to in_dim)
    
    Behavior:
        1. Run x through BatchedICNN → (…, P, out_dim)
        2. Form G=P groups of size 2 via circular pairs (i, (i+1)%P)
        3. Compute group means → (…, G, out_dim)
        4. Add static learnable shift per group
        5. Max over groups → (…, out_dim)
    """
    def __init__(self, in_dim: int, petals: int, out_dim: int = None):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.P       = petals
        self.G       = petals  # one overlapping pair per petal
        
        # Core convex petal ensemble
        self.petals = BatchedICNN(self.in_dim, self.P, self.out_dim)
        
        # Static learnable shift bias, one per group
        self.shifts = nn.Parameter(torch.zeros(self.G))
        
        # Precompute circular pair indices for gathering
        group_idxs = [[i, (i + 1) % self.P] for i in range(self.P)]
        self.register_buffer(
            "group_indices",
            torch.tensor(group_idxs, dtype=torch.long)
        )  # shape (G, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward pass with fast-path for petals==1,
        and efficient group indexing for petals > 1.
        """
        unsqueeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)
            unsqueeze = True

        out_all = self.petals(x)  # (B, S, P, D_out)
        B, S, P, D = out_all.shape

        # === Fast-path: single petal ===
        if P == 1:
            out = out_all.squeeze(2)  # (B, S, D_out)
            if unsqueeze:
                return out.squeeze(1)  # (B, D_out)
            return out

        # === Optimized group indexing ===
        # out_all: (B, S, P, D) → (B*S, P, D)
        out_2d = out_all.view(-1, P, D)

        # Gather indices as (G*2,)
        flat_indices = self.group_indices.view(-1)  # (2*G,)
        selected = out_2d.index_select(1, flat_indices)  # (B*S, 2*G, D)

        # Reshape to (B*S, G, 2, D) → mean over dim=2
        grouped = selected.view(B, S, self.G, 2, D)
        means = grouped.mean(dim=3)  # (B, S, G, D)

        # Add learnable shift and reduce
        shifted = means + self.shifts.view(1, 1, self.G, 1)
        out, _ = shifted.max(dim=2)  # (B, S, D)

        if unsqueeze:
            out = out.squeeze(1)  # (B, D)

        return out
