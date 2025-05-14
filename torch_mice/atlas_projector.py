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
