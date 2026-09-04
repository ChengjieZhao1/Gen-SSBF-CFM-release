import math
import time

import torch

from .codebook import compute_rsrp
from .noise import process_rsrp


def target_y_star(h):
    n_antennas = h.shape[1]
    w_mrt = torch.exp(1j * torch.angle(h)) / math.sqrt(n_antennas)
    y = torch.fft.fft(w_mrt, dim=-1)
    return torch.stack([y.real, y.imag], dim=1)


def y_to_beam(y):
    y_complex = y[..., 0, :] + 1j * y[..., 1, :]
    w_raw = torch.fft.ifft(y_complex, dim=-1)
    return (torch.exp(1j * torch.angle(w_raw)) / math.sqrt(w_raw.shape[-1])).to(torch.complex64)


def mrt_gain(h):
    n_antennas = h.shape[1]
    w_mrt = torch.exp(1j * torch.angle(h)) / math.sqrt(n_antennas)
    gain = (h.conj() * w_mrt).sum(dim=-1).abs() ** 2
    return gain


@torch.no_grad()
def gap_to_mrt_db(beams, h, eps=1e-12):
    if beams.ndim == 2:
        beams = beams[:, None, :]
    gain = (h.conj().unsqueeze(1) * beams).sum(dim=-1).abs() ** 2
    best = gain.max(dim=1).values
    gap = 10.0 * torch.log10((best + eps) / (mrt_gain(h) + eps))
    return {
        "mean_gap_db": float(gap.mean().item()),
        "median_gap_db": float(gap.median().item()),
        "p05_gap_db": float(torch.quantile(gap, 0.05).item()),
        "p10_gap_db": float(torch.quantile(gap, 0.10).item()),
        "within_3db": float((gap >= -3.0).float().mean().item()),
        "within_1db": float((gap >= -1.0).float().mean().item()),
    }


@torch.no_grad()
def evaluate_generator_gap(
    model,
    w_codebook,
    h_eval,
    n_candidates=8,
    n_steps=40,
    batch_size=512,
    noisy=False,
    pt_dbm=40.0,
):
    model.eval()
    beams = []
    for start in range(0, h_eval.shape[0], batch_size):
        hb = h_eval[start:start + batch_size]
        r = compute_rsrp(hb, w_codebook)
        r_shape, r_global = process_rsrp(r, noisy=noisy, pt_dbm=pt_dbm)
        y = model.sample_y(r_shape, r_global, n_candidates=n_candidates, n_steps=n_steps)
        beams.append(y_to_beam(y))
    return gap_to_mrt_db(torch.cat(beams, dim=0), h_eval)


@torch.no_grad()
def latency_ms(fn, warmup=2, repeat=5):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - start) / max(repeat, 1)
