# -*- coding: utf-8 -*-
# Copyright © 2025 Joshuah Rainstar
# License: see ../LICENSE.txt


import torch
import torch.nn as nn
import torch.nn.functional as F

# LayerNorm computes statistics (mean, std) from the input x, making its normalization step input-dependent and non-affine.
# This introduces nonlinear dependencies on x, breaking convexity due to division by x-dependent std(x).
# BatchAffineNorm uses frozen global stats and affine transformations, preserving convexity in x.

__all__ = ["BatchAffineNorm"]

class BatchAffineNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, momentum=2e-3):
        super().__init__()
        self.register_buffer('mu',     torch.zeros(dim))
        self.register_buffer('sigma',  torch.ones(dim))
        self.register_buffer('steps',  torch.tensor(0, dtype=torch.long))
        self.rho   = nn.Parameter(torch.full((dim,), -2.0))  # γ = sigmoid(ρ) ∈ (0,1)
        self.beta  = nn.Parameter(torch.zeros(dim))
        self.mom   = momentum
        self.eps = eps

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                m = x.mean(dim=(0, 1))  # mean over both batch and sequence
                v = x.var(dim=(0, 1), unbiased=False).sqrt()
                self.mu    = (1-self.mom) * self.mu    + self.mom * m
                self.sigma = (1-self.mom) * self.sigma + self.mom * v
                self.steps += 1

        γ = torch.sigmoid(self.rho)           # (0,1)
        x_hat = (x - self.mu) / (self.sigma + self.eps)
        return x_hat * γ + self.beta
