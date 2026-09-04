import math

import torch


def dbm_to_watt(dbm):
    return 10.0 ** (float(dbm) / 10.0) * 1e-3


def noise_power_watt(noise_psd_dbm_hz=-170.0, bandwidth_hz=100e6):
    return dbm_to_watt(float(noise_psd_dbm_hz) + 10.0 * math.log10(float(bandwidth_hz)))


def paper_noise_scalars(
    pt_dbm=40.0,
    shadowing_var_db=1.0,
    noise_psd_dbm_hz=-170.0,
    bandwidth_hz=100e6,
    device=None,
    dtype=torch.float32,
):
    return {
        "pt_watt": torch.tensor(dbm_to_watt(pt_dbm), device=device, dtype=dtype),
        "shadowing_var_db": torch.tensor(float(shadowing_var_db), device=device, dtype=dtype),
        "noise_watt": torch.tensor(
            noise_power_watt(noise_psd_dbm_hz, bandwidth_hz), device=device, dtype=dtype
        ),
    }


def process_rsrp(
    r,
    noisy=False,
    pt_dbm=40.0,
    shadowing_var_db=1.0,
    noise_psd_dbm_hz=-170.0,
    bandwidth_hz=100e6,
    n_ssb_symbols=5,
    common_shadowing=True,
    eps=1e-12,
):
    """Apply the paper dB-domain RSRP perturbation model and normalize.

    Input r is natural-log array gain, matching the authors' experiment scripts.
    The returned pair is (shape-normalized RSRP, global [mean, std]).
    """
    if noisy:
        scalars = paper_noise_scalars(
            pt_dbm=pt_dbm,
            shadowing_var_db=shadowing_var_db,
            noise_psd_dbm_hz=noise_psd_dbm_hz,
            bandwidth_hz=bandwidth_hz,
            device=r.device,
            dtype=r.dtype,
        )
        pt = scalars["pt_watt"]
        sh2 = scalars["shadowing_var_db"]
        n2 = scalars["noise_watt"]
        gain = torch.exp(r).clamp_min(eps)
        log10 = torch.log(torch.tensor(10.0, device=r.device, dtype=r.dtype))
        shadow_std = torch.sqrt(sh2) * log10 / 10.0
        thermal_var = (n2 ** 2 + 2.0 * n2 * pt * gain) / (
            pt ** 2 * gain ** 2 * float(n_ssb_symbols) ** 2 * 100.0
        )
        bias = n2 / pt / gain / 100.0
        if common_shadowing:
            shadow = shadow_std * torch.randn(r.shape[0], 1, device=r.device, dtype=r.dtype)
        else:
            shadow = shadow_std * torch.randn_like(r)
        r = r + bias + shadow + torch.sqrt(thermal_var.clamp_min(eps)) * torch.randn_like(r)

    mu = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True)
    return (r - mu) / (std + eps), torch.cat([mu, std], dim=1)

