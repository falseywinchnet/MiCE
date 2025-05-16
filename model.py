import torch
import torch.nn as nn

from torch_mice import (
    ConvexExpansionAttention,
    ConvexContractionAttention,
    ConvexGate,            # if you still want a gate in the future
    BatchAffineNorm,
    VectorHull,
    GeometricConvexEmbedding
)
class ConvexBlock(nn.Module):
    def __init__(self, dim: int, expansion_factor: int = 3):
        """
        A single convex block:
          1) Expansion attention:   dim -> dim * expansion_factor
          2) Norm
          3) VectorHull MLP:        dim * expansion_factor -> dim * expansion_factor
          4) Norm
          5) Contraction attention: dim * expansion_factor -> dim
          6) Norm
          7) VectorHull MLP:        dim -> dim
          8) Norm
        Residual: adds input x (shape B,L,dim) to final output (B,L,dim).
        """
        super().__init__()
        self.attn_exp = ConvexExpansionAttention(dim, dim * expansion_factor)
        self.norm1    = BatchAffineNorm(dim * expansion_factor)

        self.vh1      = VectorHull(dim * expansion_factor, petals=2,out_dim= dim * expansion_factor,invert=False)
        self.norm2    = BatchAffineNorm(dim * expansion_factor)

        self.attn_ctr = ConvexContractionAttention(dim * expansion_factor, dim)
        self.norm3    = BatchAffineNorm(dim)

        self.vh2      = VectorHull(dim, petals=2,out_dim= dim,invert=False)
        self.norm4    = BatchAffineNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, dim)
        residual = x  # will be added back at the end

        # 1) Expansion attention
        #    in:  (B, L, dim)
        #    out: (B, L, dim * 3)
        x = self.attn_exp(x)
        x = self.norm1(x)

        # 2) VectorHull MLP #1
        #    in/out: (B, L, dim * 3)
        x = self.vh1(x)
        x = self.norm2(x)

        # 3) Contraction attention
        #    in:  (B, L, dim * 3)
        #    out: (B, L, dim)
        x = self.attn_ctr(x)
        x = self.norm3(x)

        # 4) VectorHull MLP #2
        #    in/out: (B, L, dim)
        x = self.vh2(x)
        x = self.norm4(x)

        # Residual connection
        # both x and residual: (B, L, dim)
        return x + residual


class ConvexLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 3,
        num_blocks: int = 2,
        max_length: int = 512,
    ):
        """
        Convex LLM:
          – Three positive embeddings (q, k, v), each (vocab_size → embed_dim)
          – Learned causal positional embedding (max_length → embed_dim)
          – Concatenate to get initial hidden dim = 3 * embed_dim
          – Stack `num_blocks` of ConvexBlock(dim=3*embed_dim)
          – Final VectorHull decoder projecting (3*embed_dim → vocab_size)
        """
        super().__init__()
        E = embed_dim
        self.embed_q   = GeometricConvexEmbedding(vocab_size, E)
        self.embed_k   = GeometricConvexEmbedding(vocab_size, E)
        self.embed_v   = GeometricConvexEmbedding(vocab_size, E)
        self.pos_embed = nn.Embedding(max_length, E)

        block_dim = 3 * E
        self.blocks = nn.ModuleList([
            ConvexBlock(dim=block_dim, expansion_factor=3)
            for _ in range(num_blocks)
        ])

        # Final decoder: (B, L, 3E) → (B, L, V)
        self.decoder = VectorHull(block_dim, petals=2,out_dim= vocab_size,invert=False)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        input_ids: (B, L)
        returns logits: (B, L, vocab_size)
        """
        B, L = input_ids.size()
        # 1) Token embeddings (B, L, E) each
        q = self.embed_q(input_ids)
        k = self.embed_k(input_ids)
        v = self.embed_v(input_ids)

        # 2) Causal positional ids & embedding
        #    pos_ids: (1, L) → broadcast to (B, L)
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        pos    = self.pos_embed(pos_ids)  # (B, L, E)

        # 3) Inject pos embedding into each channel
        q = q + pos  # (B, L, E)
        k = k + pos
        v = v + pos

        # 4) Concatenate channels → (B, L, 3E)
        x = torch.cat([q, k, v], dim=-1)

        # 5) Pass through each ConvexBlock (dims preserved = 3E)
        for block in self.blocks:
            x = block(x)  # (B, L, 3E)

        # 6) Final decode to vocab logits
        logits = self.decoder(x)  # (B, L, V)
        return logits
