# -*- coding: utf-8 -*-
# Copyright © 2025 Joshuah Rainstar
# License: see ../LICENSE.txt

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


__all__ = ["ConvexCompressor","ConvexSimilarityHash"]

class ConvexCompressor(nn.Module):
    """
    Learns to compress a (B, T) sequence down to (B, T//2).
    """
    def __init__(self, input_len: int):
        super().__init__()
        self.output_len = input_len // 2
        self.net = nn.Sequential(
            nn.Linear(input_len, self.output_len),
            nn.SiLU(),
            nn.Linear(self.output_len, self.output_len),
            nn.Tanh()  # hard bound to (-1, 1)
        )

    def forward(self, x):  # x: (B, T)
        return self.net(x)  # (B, T//2)

        
    

class ConvexSimilarityHash(nn.Module):
    """
    1) Apply a linear taper in time.
    2) At each t, collapse the embedding vector to a phase angle:
         – radius = |x[t,0]|, starting angle = 0 or π by sign
         – accumulate arcsin(rem / radius) over the other channels
    3) Build a convex “bump” past‐kernel, apply to x[:, :, 0], but only keep odd positions → TDIFF
    4) Run the angles (PHASES) through a small learned compressor → THASH

    Output: torch.Tensor of shape (B, T//2, 2) with [THASH, TDIFF] in the last dim.
    """
    def __init__(self, embed_dim: int, seq_len: int, window_size: int = 15, eps: float = 1e-6):
        super().__init__()
        assert seq_len % 2 == 0, "seq_len must be even"
        self.E = embed_dim
        self.T = seq_len
        self.half_T = seq_len // 2
        self.window_size = window_size
        self.eps = eps

        # Step 1 taper: linear ramp from 1→0
        taper = torch.linspace(1.0, 0.0, self.T)
        self.register_buffer('taper', taper)  # (T,)

        # Step 4 compressor
        self.compressor = ConvexCompressor(self.T)
        self.phase = ConvexCompressor(self.T)

    def _compute_tdiff_phase(self, x0, window_size=15):
        """
        Computes TDIFF using causal window over x0 ∈ (B, T),
        treating each position as a point in a 1D field and accumulating
        phase-aligned, value-modulated contributions from prior steps.
    
        Output: TDIFF ∈ (B, T//2)
        """
        B, T = x0.shape
        tdiff = torch.zeros(B, T, device=x0.device)
    
        for i in range(T):
            j_start = max(i - window_size, 0)
            window = x0[:, j_start:i]  # shape (B, w), strictly before i
    
            # Weight linearly by recency (optional — can try learned weights later)
            if window.shape[1] > 0:
                offsets = torch.arange(window.shape[1], 0, -1, device=x0.device).float()  # [w, ..., 1]
                weights = offsets / offsets.sum()  # normalize
                tdiff[:, i] = (window * weights.view(1, -1)).sum(dim=1)
            else:
                tdiff[:, i] = 0.0
    
        return tdiff

    def forward(self, x):
        """
        x: (B, T, E)
        returns: (B, T//2, 2)
        """
        B, T, E = x.shape
        assert T == self.T and E == self.E

        # 1) taper
        x_t = x * self.taper.view(1, T, 1)  # (B, T, E)

        # 2) phase‐projection
        
        c0 = x_t[..., 0]                    # (B, T)
        r  = c0.abs() + self.eps            # radius
        start = torch.where(c0>=0, 0.0, math.pi).to(c0)  # (B, T)
        rem   = x_t[..., 1:]                # (B, T, E-1)
        v     = (rem / r.unsqueeze(-1)).clamp(-1+self.eps, 1-self.eps)
        alpha = torch.asin(v).sum(dim=-1)   # (B, T)
        PHASES = start + alpha              # (B, T)
    
        TDIFF = self.phase(self._compute_tdiff_phase(c0))   # (B, T//2)

        # 4) compress PHASES
        THASH = self.compressor(PHASES)       # (B, T//2)
        # stack → (B, T//2, 2)
        return torch.stack([THASH, TDIFF], dim=-1)
