# -*- coding: utf-8 -*-
# Copyright © 2025 Joshuah Rainstar
# License: see LICENSE.txt


'''
an example model.
does not yet converge beyond loss of ~3.25
important note: set loss as high as you want.
Will it learn? probably not. 
but will other models explode?
why wont this?
think about it.
--josh

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_mice import (
    BatchedICNN,
    BatchedSingleICNN,
    PositiveLinearHK,
    BatchAffineNorm,
    ConvexSimilarityHash,
    
)


import scipy.fftpack
from matplotlib import pyplot as plt
def dct_basis(L, k):
    return torch.tensor(scipy.fftpack.dct(np.eye(L), norm='ortho')[:k], dtype=torch.float32)
    

# === VAE-based Query Generator ===

class VectorFieldHyperNetwork2D(nn.Module):
    def __init__(self, in_shape: tuple[int, int], out_shape: tuple[int, int], hidden_dim=32):
        """
        Args:
            in_shape  : (H_in, W_in) — e.g. (T//2, 2)
            out_shape : (H_out, W_out) — e.g. (T, D)
        """
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.resizer = nn.AdaptiveAvgPool2d(out_shape)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, x):
        """
        x: (B, H_in, W_in)
        returns: (B, H_out, W_out)
        """
        B, H_in, W_in = x.shape
        x = x.unsqueeze(1)                          # (B, 1, H_in, W_in)
        x = self.encoder(x)                         # (B, C, H_in, W_in)
        x = self.resizer(x)                         # (B, C, H_out, W_out)
        x = self.decoder(x)                         # (B, 1, H_out, W_out)
        return x.squeeze(1)                         # (B, H_out, W_out)


def normalized_alpha_softplus_attention(Q, K, V, alpha=1.5, tau=0.2, eps=1e-6, log=True):
    # 1. Scaled dot product
    logits = torch.einsum("bqd,bkd->bqk", Q, K)  # both are (B, L, D)
    logits = logits / math.sqrt(Q.size(-1))       # scale by √D

    scores = F.softplus((alpha - 1) * logits - tau) ** (1 / (alpha - 1))
    # 4. Normalize manually (row-wise softmax)
    weights = scores / (scores.sum(dim=-1, keepdim=True) + eps)
    attn_score = weights.sum(dim=2)  # (B, K)

    # Normalize each sample separately (min-max per row)
    min_vals = attn_score.min(dim=-1, keepdim=True).values
    max_vals = attn_score.max(dim=-1, keepdim=True).values
    attn_score = (attn_score - min_vals) / (max_vals - min_vals + 1e-6)  # (B, K)
    

    # 5. Apply attention
    return torch.matmul(weights, V),attn_score
    
class PerceptualAttentionBlock(nn.Module):
    def __init__(self, model_dim, seq_len):
        super().__init__()
        self.hash = ConvexSimilarityHash(model_dim, seq_len)
        self.query_gen = VectorFieldHyperNetwork2D(in_shape=(seq_len//2, 2), out_shape=(seq_len, model_dim))
        self.key_gen   = VectorFieldHyperNetwork2D(in_shape=(seq_len//2, 2), out_shape=(seq_len, model_dim))
        self.value_gen   = VectorFieldHyperNetwork2D(in_shape=(seq_len//2, 2), out_shape=(seq_len, model_dim))

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)

        
    def forward(self, x):
        """
        x: (B, L, D) — used as Q input, K input, and V source
        Returns:
            attention output: (B, L, D)
            kl divergence from VAE
        """
        h = self.hash(x)                # (B, T//2,2)
        Q = self.query_gen(h)    # (B, L, D)
        K = self.key_gen(h)            # (B, L, D)
        V = self.value_gen(h)
        Q = self.q_proj(Q)
        K = self.k_proj(K)
        V = self.v_proj(V)
        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)
        V = F.normalize(V, dim=-1)
       
        attn_out,attn_score = normalized_alpha_softplus_attention(Q, K, V)
        return attn_out, 0 ,attn_score


        
class ConvexBlock(nn.Module):
    def __init__(self, model_dim: int, seq_len: int):
        super().__init__()
        self.attn  = PerceptualAttentionBlock(model_dim, seq_len)
        self.norm1 = BatchAffineNorm(model_dim)
        self.icnn1 = BatchedSingleICNN(in_dim=model_dim, out_dim=model_dim)
        self.norm2 = BatchAffineNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, dim)
        att,kl, attn_score= self.attn(self.norm1(x))
        x = x + att       # → (B, L, dim)

        # ICNN #1 + norm
        x = x + self.icnn1(self.norm2(x))          # → (B, L, dim)

        # residual add
        return x,kl , attn_score                      # → (B, L, dim)


class ConvexLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 128,
        seq_len: int = 128,
        num_blocks: int = 6,
    ):
        super().__init__()
        self.embedding = GeometricConvexEmbedding(vocab_size, model_dim)
        self.wpe = nn.Embedding(seq_len, model_dim)
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([
            ConvexBlock(model_dim, seq_len)
            for _ in range(num_blocks)
        ])

        self.decoder = BatchedSingleICNN(in_dim=model_dim, out_dim=vocab_size)

        

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        input_ids: (B, L)
        returns   : logits (B, L, vocab_size)
        """
        # -- Embed
    
        x = self.embedding(input_ids)            # (B, L, E)
        B,t,_ = x.shape
 
        # -- Convex blocks
        kl = 0.0
        attn_scores = []
        for block in self.blocks:
            x, klt ,attn_score = block(x)                         # (B, L, E)
            kl += klt
            attn_scores.append(attn_score)

        # -- Decode
        attn_vis = torch.stack(attn_scores).mean(dim=0)
        logits = self.decoder(x)                 # (B, L, V)
        return logits,kl/self.num_blocks,attn_vis
