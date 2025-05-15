# -*- coding: utf-8 -*-
# Copyright © 2025 Joshuah Rainstar
# License: see ../LICENSE.txt

import torch
import torch.nn as nn

from .batched_icnn import BatchedICNN
from .atlas_projector import AtlasProjector

__all__ = ["VectorHull"]

"""
    VectorHull: Convex vector‐valued function via overlapping shifted max‐of‐means fusion,
    with optional exact inversion of fixed affine projections.

    Pipeline:
      1. Project x into P fixed, orthonormal petal coordinates: z_p = A_p x
      2. Evaluate each convex subnetwork (BatchedICNN) on z_p
      3. (Optional) Reproject each petal’s output back to the global frame: v_p = A_pᵀ f_p(z_p)
      4. Group the P outputs into P overlapping pairs (i, i+1 mod P)
      5. Compute the mean over each pair, add a learnable shift bias
      6. Compute the coordinate‐wise max over all groups → final output y(x)

    This structure guarantees that each chart R_p ∘ f_p ∘ A_p is convex, and the max‐fusion
    preserves convexity.  The `invert` flag toggles step 3 (the exact inversion) on or off.
    """


class VectorHull(nn.Module):
    """
    VectorHull with optional exact inversion of fixed projections.

    Args:
        in_dim:  input feature dim D
        petals:  number of petals P
        out_dim: output dim per petal (defaults to in_dim)
        invert:  if True, apply exact inversion A_p^{-1} to petal outputs before fusion
    """
    def __init__(self, in_dim: int, petals: int, out_dim: int = None, invert: bool = True):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.P       = petals
        self.G       = petals
        self.invert  = invert

        # 1) fixed projections (with .inverse available)
        self.projector = AtlasProjector(in_dim, petals)

        # 2) Batched ICNN expecting pre-projected + flat inputs
        self.petals_net = BatchedICNN(in_dim, petals, self.out_dim)

        # 3) convex‐hull fusion params
        self.shifts = nn.Parameter(torch.zeros(self.G))
        group_idxs = [[i, (i + 1) % self.P] for i in range(self.P)]
        self.register_buffer("group_indices", torch.tensor(group_idxs, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ensure shape (B, S, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, S, D = x.shape
        N = B * S

        # flatten batch+seq to (N,D)
        x_flat = x.reshape(N, D)

        # forward‐project into each petal chart → (P, N, D)
        x_proj = self.projector(x_flat)

        # petal ICNN: returns (N, P, out_dim)
        f_p = self.petals_net(x_proj, x_flat)

        # if inversion requested, bring each petal output back to global frame
        if self.invert:
            assert self.out_dim == self.in_dim, "Cannot invert unless out_dim == in_dim; projection is only defined on input space"
            # permute → (P, N, out_dim)
            f_p = f_p.permute(1, 0, 2)
            # inverse‐project → (N, P, out_dim)
            f_p = self.projector.inverse(f_p)

        # reshape for fusion: (B, S, P, out_dim)
        f_p = f_p.reshape(B, S, self.P, self.out_dim)
        out_2d = f_p.reshape(N, self.P, self.out_dim)

        # grouping & convex‐hull max‐fusion (identical in both modes)
        flat_inds = self.group_indices.view(-1)         # (2*G,)
        selected  = out_2d.index_select(1, flat_inds)    # (N,2G,out_dim)
        grouped   = selected.view(N, self.G, 2, self.out_dim)
        means     = grouped.mean(dim=2)                 # (N,G,out_dim)
        if not self.invert: #apply an approximating shift - Rainstar
            means   = means + self.shifts.view(1, self.G, 1)
        hull_out, _ = means.max(dim=1)                # (N,out_dim)

        # restore (B,S,out_dim)
        return hull_out.reshape(B, S, self.out_dim)



