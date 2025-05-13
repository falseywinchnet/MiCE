class BatchedICNN(nn.Module):
    """
    Input-Convex Neural Network over batches of points,
    with additive convex gating to guarantee convexity.
    """
    def __init__(self, in_dim: int, petals: int, out_dim: int):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.P       = petals
        D = in_dim
        D_out = out_dim

        self.d1 = 2 * D
        self.d2 = D_out  # instead of D

        self.layer0   = PositiveLinear3DHK(petals, D, self.d1)
        self.layer1   = PositiveLinear3DHK(petals, self.d1, self.d2)
        self.res_proj = PositiveLinear3DHK(petals, 2 * D, self.d2)

        self.gate0_net = ConvexGate(D, self.d1)
        self.gate1_net = ConvexGate(D, self.d2)
        self.extra_gate0_nets = nn.ModuleList([
            ConvexGate(D, self.d1) for _ in range(self.P)
        ])
        self.extra_gate1_nets = nn.ModuleList([
            ConvexGate(D, self.d2) for _ in range(self.P)
        ])

        self.out_bias = nn.Parameter(torch.zeros(petals, self.d2))
        self.act      = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_dim)        # (N, D)
        N      = x_flat.size(0)

        # compute vector gates and expand over petals
        g0 = self.gate0_net(x_flat)               # (N, d1)
        g0 = g0.unsqueeze(0).expand(self.P, N, self.d1)
        g1 = self.gate1_net(x_flat)               # (N, d2)
        g1 = g1.unsqueeze(0).expand(self.P, N, self.d2)

        # petal inputs (use in_dim, not d2)
        x_in = x_flat.unsqueeze(0).expand(self.P, N, self.in_dim)  # (P, N, D)

        # layer0 + additive gating
        z0 = self.layer0(x_in)                     # (P, N, d1)
        z0 = self.act(z0 + g0)                     # (P, N, d1)

        # extra per-petal additive gates
        extra0 = torch.stack([g(x_flat) for g in self.extra_gate0_nets], dim=0)
        z0 = self.act(z0 + extra0)                 # (P, N, d1)

        # layer1 + additive gating
        z1 = self.layer1(z0)                       # (P, N, d2)
        z1 = self.act(z1 + g1)                     # (P, N, d2)

        extra1 = torch.stack([g(x_flat) for g in self.extra_gate1_nets], dim=0)
        z1 = self.act(z1 + extra1)                 # (P, N, d2)

        # residual path (use in_dim for expansion)
        res_in = x_flat.unsqueeze(0).expand(self.P, N, self.in_dim)  # (P, N, D)
        res_in = torch.cat([res_in, res_in], dim=-1)  # (P, N, 2*D)
        res    = self.res_proj(res_in)                # (P, N, d2)

        # combine + bias
        out = self.act(z1 + res) + self.out_bias.unsqueeze(1)  # (P, N, d2)

        # reshape back to original batch dims + petals + D_out
        out = out.permute(1, 0, 2)  # (N, P, d2)
        new_shape = list(orig_shape[:-1]) + [self.P, self.out_dim]
        return out.reshape(new_shape)
