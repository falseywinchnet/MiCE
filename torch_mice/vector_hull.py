class VectorHull(nn.Module):
    """
    Convex vector-valued function using overlapping shifted max-of-means.
    Each petal appears in two groups (pairs), and group outputs are shift-biased.
    """
    def __init__(self, in_dim: int, petals: int, out_dim: int = None):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.petals  = BatchedICNN(self.in_dim, petals, self.out_dim)
        self.P       = petals
        self.G       = petals  # each group is (i, i+1) mod P → P groups
        self.shifts  = nn.Parameter(torch.zeros(self.G))  # one shift per overlapping group

        # Precompute static index mapping: (G, 2)
        group_indices = []
        for i in range(self.P):
            group_indices.append([i, (i + 1) % self.P])  # wrap-around
        self.register_buffer('group_indices', torch.tensor(group_indices))  # (G, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
          """
          x: (..., D) → returns: (..., D_out)
          Supports both (B, D) and (B, S, D)
          """
          unsqueezed = False
          if x.dim() == 2:
              x = x.unsqueeze(1)  # (B, 1, D)
              unsqueezed = True

          out_all = self.petals(x)               # (B, S, P, D)
          B, S, P, D = out_all.shape
          G = self.G

          # Grouping logic (unchanged)
          idx = self.group_indices.view(1, 1, G, 2, 1).expand(B, S, G, 2, D)
          petal_expanded = out_all.unsqueeze(2).expand(B, S, G, P, D)
          grouped = torch.gather(petal_expanded, dim=3, index=idx)  # (B, S, G, 2, D)
          means = grouped.mean(dim=3)            # (B, S, G, D)
          shifts = self.shifts.view(1, 1, G, 1)  # (1, 1, G, 1)
          shifted = means + shifts               # (B, S, G, D)
          out, _ = shifted.max(dim=2)            # (B, S, D)

          if unsqueezed:
              out = out.squeeze(1)               # restore shape (B, D)
          return out
