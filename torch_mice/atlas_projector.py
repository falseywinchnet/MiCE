import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AtlasProjector"]


class AtlasProjector(nn.Module):
    def __init__(self, in_dim: int, petals: int):
        super().__init__()
        self.in_dim = in_dim
        self.petals = petals
        A = self._build_projections(in_dim, petals)  # (P, D, D)
        self.register_buffer('A', A)                 # Forward projection
        self.register_buffer('A_inv', A.transpose(1, 2))  # Inverse (orthonormal)

    def _build_projections(self, D, P):
        base = torch.eye(D)
        Q, _ = torch.linalg.qr(base + 1e-3 * torch.randn(D, D))
        rotations = []
        for p in range(P):
            perm = torch.roll(torch.arange(D), shifts=p).tolist()
            rot = Q[:, perm]
            rotations.append(rot)
        return torch.stack(rotations, dim=0)  # (P, D, D)

    def forward(self, x):
        # x: (N, D)
        # return projected input (P, N, D)
        return torch.einsum('pdi,ni->pnd', self.A, x)

    def inverse(self, z):
        # z: (P, N, D)
        # return unprojected (N, P, D)
        return torch.einsum('pij,pnj->npi', self.A_inv, z)
