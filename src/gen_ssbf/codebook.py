import math

import torch
import torch.nn as nn

from .noise import process_rsrp


class ProbingCodebook(nn.Module):
    """Constant-modulus analog probing codebook parameterized by phase."""

    def __init__(self, k, n_antennas, seed=0):
        super().__init__()
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        phi = (2.0 * math.pi) * torch.rand((k, n_antennas), generator=g) - math.pi
        self.phi = nn.Parameter(phi)

    def forward(self):
        k, n = self.phi.shape
        return (torch.exp(1j * self.phi) / math.sqrt(n)).transpose(0, 1).contiguous()


def compute_rsrp(h, w, eps=1e-12):
    y = h.conj() @ w
    return torch.log(y.real ** 2 + y.imag ** 2 + eps)


def normalize_rsrp(r, eps=1e-6):
    x = r - r.mean(dim=1, keepdim=True)
    return x / (r.std(dim=1, keepdim=True) + eps)


def covariance_matrix(x, eps=1e-3):
    b, k = x.shape
    xc = x - x.mean(dim=0, keepdim=True)
    eye = torch.eye(k, device=x.device, dtype=x.dtype)
    return (xc.T @ xc) / max(b - 1, 1) + eps * eye


def cov_logdet_loss(r_norm, eps=1e-3):
    cov = covariance_matrix(r_norm, eps=eps)
    sign, log_abs_det = torch.linalg.slogdet(cov)
    penalty = torch.where(sign > 0, torch.zeros_like(log_abs_det), torch.full_like(log_abs_det, 100.0))
    return -(log_abs_det - penalty)


def cov_logdet_value(r_norm, eps=1e-3):
    cov = covariance_matrix(r_norm, eps=eps)
    eigvals = torch.linalg.eigvalsh(cov)
    return torch.log(eigvals).sum().item()


def corr_penalty(w):
    gram = w.conj().T @ w
    gram_abs2 = gram.real ** 2 + gram.imag ** 2
    return gram_abs2.sum() - gram_abs2.diag().sum()


def coverage_loss(r, threshold=None, temperature=0.2):
    if threshold is None:
        threshold = torch.quantile(r.detach().amax(dim=1), 0.10)
    smooth_max = temperature * torch.logsumexp(r / temperature, dim=1)
    return torch.relu(threshold - smooth_max).mean()


@torch.no_grad()
def eig_metrics(w, h, noisy=False, pt_dbm=40.0):
    r = compute_rsrp(h, w)
    if noisy:
        r_norm, _ = process_rsrp(r, noisy=True, pt_dbm=pt_dbm)
    else:
        r_norm = normalize_rsrp(r)
    cov = covariance_matrix(r_norm)
    eigvals = torch.linalg.eigvalsh(cov)
    lmin = eigvals.min().item()
    lmax = eigvals.max().item()
    return {
        "logdet": torch.log(eigvals).sum().item(),
        "lambda_min": lmin,
        "condition_number": lmax / max(lmin, 1e-12),
    }


def train_sim_codebook(
    h_train,
    k,
    n_antennas=None,
    epochs=150,
    batch_size=1024,
    lr=3e-3,
    w_info=1.0,
    w_corr=0.1,
    w_cov=0.0,
    noisy=False,
    pt_dbm=40.0,
    seed=0,
):
    n_antennas = int(n_antennas or h_train.shape[1])
    torch.manual_seed(seed)
    model = ProbingCodebook(k, n_antennas, seed=seed).to(h_train.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    n = h_train.shape[0]
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, device=h_train.device)
        accum = {"loss": 0.0, "info": 0.0, "corr": 0.0, "coverage": 0.0, "batches": 0}
        for start in range(0, n, batch_size):
            h_batch = h_train[order[start:start + batch_size]]
            opt.zero_grad(set_to_none=True)
            w = model()
            r = compute_rsrp(h_batch, w)
            if noisy:
                r_norm, _ = process_rsrp(r, noisy=True, pt_dbm=pt_dbm)
            else:
                r_norm = normalize_rsrp(r)
            info = cov_logdet_loss(r_norm)
            corr = corr_penalty(w)
            cov = coverage_loss(r)
            loss = w_info * info + w_corr * corr + w_cov * cov
            loss.backward()
            opt.step()
            accum["loss"] += float(loss.detach().item())
            accum["info"] += float(info.detach().item())
            accum["corr"] += float(corr.detach().item())
            accum["coverage"] += float(cov.detach().item())
            accum["batches"] += 1
        denom = max(accum.pop("batches"), 1)
        history.append({"epoch": epoch, **{key: value / denom for key, value in accum.items()}})
    return model().detach(), history


@torch.no_grad()
def beam_pattern_db(w, n_grid=721):
    n_ant, k = w.shape
    spatial = torch.linspace(-1.0, 1.0, n_grid, device=w.device)
    n = torch.arange(n_ant, device=w.device, dtype=torch.float32)
    steering = torch.exp(1j * math.pi * spatial[:, None] * n[None, :]) / math.sqrt(n_ant)
    response = (steering.conj() @ w).abs() ** 2
    response = response / response.max(dim=0, keepdim=True).values.clamp_min(1e-12)
    return spatial.detach().cpu().numpy(), (10.0 * torch.log10(response.clamp_min(1e-12))).detach().cpu().numpy()
