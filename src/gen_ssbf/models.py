import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, global_dim, hidden_dim):
        super().__init__()
        self.to_gamma = nn.Linear(global_dim, hidden_dim)
        self.to_beta = nn.Linear(global_dim, hidden_dim)

    def forward(self, hidden, global_feature):
        return self.to_gamma(global_feature) * hidden + self.to_beta(global_feature)


class FiLMBlock(nn.Module):
    def __init__(self, in_dim, out_dim, cond_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.to_gamma = nn.Linear(cond_dim, out_dim)
        self.to_beta = nn.Linear(cond_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x, cond):
        hidden = self.norm(self.linear(x))
        hidden = self.to_gamma(cond) * hidden + self.to_beta(cond)
        return self.act(hidden)


class ShapeEncoder(nn.Module):
    def __init__(self, k, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class GlobalEncoder(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class ConditionEncoderFiLM(nn.Module):
    def __init__(self, k, cond_dim=128):
        super().__init__()
        self.shape_encoder = ShapeEncoder(k, cond_dim)
        self.global_encoder = GlobalEncoder(2, 128)
        self.film = FiLM(128, cond_dim)

    def forward(self, r_shape, r_global):
        return self.film(self.shape_encoder(r_shape), self.global_encoder(r_global))


class VectorField(nn.Module):
    def __init__(self, n_antennas, k, cond_dim=128, width=512, depth=3):
        super().__init__()
        self.n_antennas = int(n_antennas)
        self.cond_encoder = ConditionEncoderFiLM(k, cond_dim)
        blocks = []
        in_dim = 2 * self.n_antennas + 1
        for _ in range(depth):
            blocks.append(FiLMBlock(in_dim, width, cond_dim))
            in_dim = width
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Linear(in_dim, 2 * self.n_antennas)

    def forward(self, y, t, r_shape, r_global):
        batch = y.shape[0]
        hidden = torch.cat([y.reshape(batch, 2 * self.n_antennas), t[:, None]], dim=1)
        cond = self.cond_encoder(r_shape, r_global)
        for block in self.blocks:
            hidden = block(hidden, cond)
        return self.out(hidden).view(batch, 2, self.n_antennas)


class CFM(nn.Module):
    def __init__(self, n_antennas, k, sigma0=1.0, temperature=0.5, cond_dim=128, width=512, depth=3):
        super().__init__()
        self.n_antennas = int(n_antennas)
        self.sigma0 = float(sigma0)
        self.temperature = float(temperature)
        self.vector_field = VectorField(n_antennas, k, cond_dim=cond_dim, width=width, depth=depth)

    def loss(self, r_shape, r_global, y_star):
        batch = r_shape.shape[0]
        t = torch.rand(batch, device=r_shape.device)
        y0 = torch.randn_like(y_star) * self.sigma0
        yt = (1.0 - t)[:, None, None] * y0 + t[:, None, None] * y_star
        target_v = y_star - y0
        return F.mse_loss(self.vector_field(yt, t, r_shape, r_global), target_v)

    @torch.no_grad()
    def sample_y(self, r_shape, r_global, n_candidates=8, n_steps=40):
        batch = r_shape.shape[0]
        dt = 1.0 / float(n_steps)
        y = torch.randn(
            batch,
            n_candidates,
            2,
            self.n_antennas,
            device=r_shape.device,
            dtype=r_shape.dtype,
        ) * (self.sigma0 * self.temperature)
        rs = r_shape[:, None, :].expand(batch, n_candidates, -1).reshape(batch * n_candidates, -1)
        rg = r_global[:, None, :].expand(batch, n_candidates, -1).reshape(batch * n_candidates, -1)
        for step in range(n_steps):
            t = torch.full((batch * n_candidates,), step / float(n_steps), device=r_shape.device, dtype=r_shape.dtype)
            v = self.vector_field(y.reshape(batch * n_candidates, 2, self.n_antennas), t, rs, rg)
            y = y + dt * v.view(batch, n_candidates, 2, self.n_antennas)
        return y
