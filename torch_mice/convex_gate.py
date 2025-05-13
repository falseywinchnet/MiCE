class ConvexGate(nn.Module):
    """
    Convex & bounded gate: g(x) = 1 - exp(-softplus(Wx + b)) ∈ (0,1)^out_dim
    """
    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=True)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim)
        u = self.softplus(self.lin(x))       # (N, out_dim), convex ≥ 0
        return 1.0 - torch.exp(-u)           # (N, out_dim), convex ∈ (0,1)
