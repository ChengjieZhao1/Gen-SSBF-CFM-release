import torch

from .codebook import compute_rsrp
from .metrics import target_y_star
from .noise import process_rsrp


def _autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return torch.amp.autocast("cpu", enabled=False)


def _grad_scaler(device):
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return torch.amp.GradScaler("cpu", enabled=False)


def train_cfm(
    model,
    w_codebook,
    h_train,
    h_val=None,
    epochs=120,
    batch_size=256,
    lr=2e-4,
    weight_decay=1e-4,
    noisy=False,
    pt_dbm=40.0,
    eval_every=0,
    evaluator=None,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = _grad_scaler(h_train.device)
    r_train = compute_rsrp(h_train, w_codebook)
    history = []
    n = h_train.shape[0]
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, device=h_train.device)
        total = 0.0
        batches = 0
        model.train()
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            r_shape, r_global = process_rsrp(r_train[idx], noisy=noisy, pt_dbm=pt_dbm)
            y_star = target_y_star(h_train[idx])
            opt.zero_grad(set_to_none=True)
            with _autocast_context(h_train.device):
                loss = model.loss(r_shape, r_global, y_star)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach().item())
            batches += 1
        row = {"epoch": epoch, "loss": total / max(batches, 1)}
        if evaluator and eval_every and epoch % eval_every == 0 and h_val is not None:
            row.update(evaluator(model, h_val))
        history.append(row)
    return history
