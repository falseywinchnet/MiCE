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
        x: (..., D) or (B, S, D)
        returns: (..., out_dim)
        """
        # Handle optional sequence dim
        unsqueeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)
            unsqueeze = True

        out_all = self.petals(x)               # (B, S, P, D_out)
        B, S, P, D = out_all.shape
        G = self.G

        # Gather each overlapping pair: (B, S, G, 2, D_out)
        idx = self.group_indices.view(1, 1, G, 2, 1).expand(B, S, G, 2, D)
        expanded = out_all.unsqueeze(2).expand(B, S, G, P, D)
        grouped = torch.gather(expanded, dim=3, index=idx)

        # Mean within each group and add static shift
        means = grouped.mean(dim=3)            # (B, S, G, D_out)
        shifts = self.shifts.view(1, 1, G, 1)  # (1, 1, G, 1)
        shifted = means + shifts               # (B, S, G, D_out)

        # Max over groups
        out, _ = shifted.max(dim=2)            # (B, S, D_out)

        # Restore original shape if needed
        if unsqueeze:
            out = out.squeeze(1)               # (B, D_out)
        return out
