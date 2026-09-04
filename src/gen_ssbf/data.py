from pathlib import Path

import numpy as np
import torch


def _import_deepmimo():
    try:
        import deepmimo as dm
    except ImportError as exc:
        raise ImportError(
            "DeepMIMO is required for paper-data runs. Install deepmimo>=4.0.0b11 "
            "before running the paper reproduction scripts."
        ) from exc
    return dm


def set_deepmimo_folder(scenarios_dir):
    dm = _import_deepmimo()
    if scenarios_dir:
        dm.config.set("scenarios_folder", str(Path(scenarios_dir)))
    return dm


def load_deepmimo_channels(
    scenario,
    scenarios_dir,
    n_antennas=64,
    tx_set=4,
    tx_indices=(0,),
    rx_set=0,
    row_start=750,
    row_stop=1200,
    n_cols=180,
    max_paths=15,
    bandwidth_hz=100e6,
    max_ues=None,
    normalize=False,
    device=None,
):
    """Load the DeepMIMO O1/O1B subset used by the paper."""
    dm = set_deepmimo_folder(scenarios_dir)
    rx_indices = np.arange(row_start * n_cols, row_stop * n_cols)
    dataset = dm.load(
        scenario,
        tx_sets={int(tx_set): list(tx_indices)},
        rx_sets={int(rx_set): rx_indices},
        max_paths=int(max_paths),
    )
    params = dm.ChannelParameters()
    params.bs_antenna.shape = np.array([int(n_antennas), 1])
    params.ue_antenna.shape = np.array([1, 1])
    params.ofdm.bandwidth = float(bandwidth_hz)
    h = np.asarray(dataset.compute_channels(params)).squeeze()
    if h.ndim == 1:
        h = h[None, :]

    valid = np.ones(h.shape[0], dtype=bool)
    channel_attr = getattr(dataset, "channel", None)
    if channel_attr is not None:
        try:
            channel_valid = np.array([not (np.asarray(x) == 0).all() for x in channel_attr])
            if channel_valid.shape[0] == valid.shape[0]:
                valid &= channel_valid
        except TypeError:
            pass
    valid &= np.isfinite(h.real).all(axis=1)
    valid &= np.isfinite(h.imag).all(axis=1)
    valid &= np.linalg.norm(h, axis=1) > 1e-12
    h = h[valid].astype(np.complex64)

    if max_ues is not None:
        h = h[: int(max_ues)]
    if normalize:
        h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)

    out = torch.from_numpy(h)
    if device is not None:
        out = out.to(device)
    return out.to(torch.complex64)


def split_train_val_test(h, split_seed=7, train_fraction=0.6, val_fraction=0.2, max_test=2000):
    n = h.shape[0]
    from sklearn.model_selection import train_test_split

    indices = np.arange(n)
    holdout_fraction = 1.0 - float(train_fraction)
    train_np, holdout_np = train_test_split(indices, test_size=holdout_fraction, random_state=split_seed)
    test_fraction_within_holdout = (1.0 - float(train_fraction) - float(val_fraction)) / holdout_fraction
    val_np, test_np = train_test_split(
        holdout_np,
        test_size=test_fraction_within_holdout,
        random_state=split_seed,
    )
    train_idx = torch.as_tensor(train_np, device=h.device, dtype=torch.long)
    val_idx = torch.as_tensor(val_np, device=h.device, dtype=torch.long)
    test_idx = torch.as_tensor(test_np, device=h.device, dtype=torch.long)
    if max_test is not None:
        test_idx = test_idx[: int(max_test)]
    return h[train_idx], h[val_idx], h[test_idx]

