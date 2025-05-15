# -*- coding: utf-8 -*-
# Copyright © 2025 Joshuah Rainstar
# License: see ../LICENSE.txt


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AtlasProjector","SingleStiefelProjector","SmoothStiefelProjector"]


class AtlasProjector(nn.Module):
    def __init__(self, in_dim: int, petals: int):
        super().__init__()
        self.in_dim = in_dim
        self.petals = petals
        A = self._build_projections(in_dim, petals)  # (P, D, D)
        self.register_buffer('A', A)                 # Forward projection
        self.register_buffer('A_inv', A.transpose(1, 2))  # Inverse (orthonormal)

    def _build_projections(self,D, P, theta=math.pi / 4):
        """
        Generate P smooth, deterministic D×D orthogonal projection matrices along a geodesic in SO(D)
        using exponential map of a skew-symmetric generator.
        """
        G = torch.zeros(D, D)
        
        # Create a deterministic skew-symmetric generator in multiple planes
        for i in range(0, D-1, 2):
            G[i, i+1] = -theta
            G[i+1, i] = theta

        projections = []
        for p in range(P):
            t = p / max(P - 1, 1)  # normalized in [0, 1]
            A = torch.matrix_exp(t * G)  # lie-algebra interpolation
            projections.append(A)

        return torch.stack(projections, dim=0)  # (P, D, D)

    def forward(self, x):
        # x: (N, D)
        # return projected input (P, N, D)
        return torch.einsum('pdi,ni->pnd', self.A, x)

    def inverse(self, z):
        # z: (P, N, D)
        # return unprojected (N, P, D)
        return torch.einsum('pij,pnj->npi', self.A_inv, z)


class SingleStiefelProjector(nn.Module):
    def __init__(self, dim: int, t: float = 1.0, seed_angle: float = math.pi / 8):
        """
        Smooth full-space rotation matrix via Lie algebra exponential map.
        A = exp(t·G), where G ∈ so(D) is skew-symmetric and full-rank.
        """
        super().__init__()
        self.dim = dim
        G = self._make_skew_symmetric_generator(dim, seed_angle)
        A = torch.matrix_exp(t * G)
        self.register_buffer("A", A)         # A ∈ SO(D)
        self.register_buffer("A_inv", A.T)   # A.T = A⁻¹

    def _make_skew_symmetric_generator(self, D, angle_scale):
        G = torch.randn(D, D)
        G = G - G.T                          # skew-symmetric: Gᵀ = -G
        norm = torch.norm(G, p='fro')
        return G * (angle_scale / norm)     # scale to control rotation strength

    def forward(self, x):
        """
        Project: x ∈ (B, D) or (B, ..., D) → x @ Aᵀ
        """
        return F.linear(x, self.A)          # A @ xᵀ

    def inverse(self, y):
        """
        Inverse: y @ A⁻¹ = y @ Aᵀ
        """
        return F.linear(y, self.A_inv)

class SmoothStiefelProjector(nn.Module):
    def __init__(self, dim: int, petals: int, seed_angle: float = math.pi / 8):
        """
        petal-dim version:
        builds P rotations A_p = exp(t_p · G) with t_p = p/(P-1)
        """
        super().__init__()
        self.dim = dim
        self.petals = petals

        # build one full-rank G
        G = self._make_skew_symmetric_generator(dim, seed_angle)

        # stack P rotations at t = [0,1] along geodesic
        As = []
        for p in range(petals):
            t = p / max(petals - 1, 1)
            As.append(torch.matrix_exp(t * G))
        A = torch.stack(As, dim=0)  # shape: (P, D, D)

        self.register_buffer("A", A)
        self.register_buffer("A_inv", A.transpose(1, 2))

    def _make_skew_symmetric_generator(self, D, angle_scale):
        G = torch.randn(D, D)
        G = G - G.T
        norm = torch.norm(G, p="fro")
        return G * (angle_scale / norm)

    def forward(self, x):
        """
        x: (N, D) → returns (P, N, D) with each A_p @ xᵀ
        """
        # A: [P, D, D], x: [N, D] → out: [P, N, D]
        return torch.einsum("pdi,ni->pnd", self.A, x)

    def inverse(self, z):
        """
        z: (P, N, D) → returns (N, P, D) with A_p⁻¹ @ z_p
        """
        # A_inv: [P, D, D], z: [P, N, D] → [N, P, D]
        return torch.einsum("pij,pnj->npi", self.A_inv, z)
