import argparse
from pathlib import Path

import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from gen_ssbf.codebook import eig_metrics, train_sim_codebook
from gen_ssbf.data import load_deepmimo_channels, split_train_val_test
from gen_ssbf.metrics import evaluate_generator_gap, latency_ms
from gen_ssbf.models import CFM
from gen_ssbf.plots import plot_codebook_loss, plot_gap_summary, plot_training_curve
from gen_ssbf.training import train_cfm
from gen_ssbf.utils import ensure_parent, resolve_device, seed_all, write_json


def load_codebook(path, device):
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "w_codebook" in payload:
        payload = payload["w_codebook"]
    return payload.to(device=device, dtype=torch.complex64)


def main():
    parser = argparse.ArgumentParser(description="Train the FiLM-CFM beam generator.")
    parser.add_argument("--scenarios-dir", required=True)
    parser.add_argument("--scenario", default="o1_28")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--n-antennas", type=int, default=64)
    parser.add_argument("--codebook", default=None)
    parser.add_argument("--codebook-epochs", type=int, default=300)
    parser.add_argument("--cfm-epochs", type=int, default=120)
    parser.add_argument("--batch-size-codebook", type=int, default=1024)
    parser.add_argument("--batch-size-cfm", type=int, default=256)
    parser.add_argument("--coverage-weight", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--cond-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n-steps", type=int, default=40)
    parser.add_argument("--noisy", action="store_true")
    parser.add_argument("--pt-dbm", type=float, default=40.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--max-ues", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=2000)
    parser.add_argument("--save-codebook", default="results/sim_codebook.pt")
    parser.add_argument("--save-model", default="results/film_cfm.pt")
    parser.add_argument("--out", default="results/cfm_summary.json")
    parser.add_argument("--plot-dir", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    seed_all(args.seed)
    h = load_deepmimo_channels(
        args.scenario,
        args.scenarios_dir,
        n_antennas=args.n_antennas,
        max_ues=args.max_ues,
        device=device,
    )
    h_train, h_val, h_test = split_train_val_test(h, split_seed=args.split_seed, max_test=args.max_test)

    codebook_history = None
    if args.codebook:
        w_codebook = load_codebook(args.codebook, device)
        expected_shape = (args.n_antennas, args.K)
        if tuple(w_codebook.shape) != expected_shape:
            raise ValueError(f"Expected codebook shape {expected_shape}, got {tuple(w_codebook.shape)}")
        codebook_path = Path(args.codebook)
    else:
        w_codebook, codebook_history = train_sim_codebook(
            h_train,
            args.K,
            epochs=args.codebook_epochs,
            batch_size=args.batch_size_codebook,
            w_cov=args.coverage_weight,
            noisy=args.noisy,
            pt_dbm=args.pt_dbm,
            seed=args.seed,
        )
        codebook_path = ensure_parent(args.save_codebook)
        torch.save({"w_codebook": w_codebook.detach().cpu(), "settings": vars(args)}, codebook_path)

    model = CFM(
        args.n_antennas,
        args.K,
        temperature=args.temperature,
        cond_dim=args.cond_dim,
        width=args.width,
        depth=args.depth,
    ).to(device)

    def evaluator(current_model, h_eval):
        stats = evaluate_generator_gap(
            current_model,
            w_codebook,
            h_eval,
            n_candidates=args.M,
            n_steps=args.n_steps,
            noisy=args.noisy,
            pt_dbm=args.pt_dbm,
        )
        return {f"val_{key}": value for key, value in stats.items()}

    cfm_history = train_cfm(
        model,
        w_codebook,
        h_train,
        h_val=h_val,
        epochs=args.cfm_epochs,
        batch_size=args.batch_size_cfm,
        noisy=args.noisy,
        pt_dbm=args.pt_dbm,
        eval_every=args.eval_every,
        evaluator=evaluator,
    )
    test_gap = evaluate_generator_gap(
        model,
        w_codebook,
        h_test,
        n_candidates=args.M,
        n_steps=args.n_steps,
        noisy=args.noisy,
        pt_dbm=args.pt_dbm,
    )
    h_lat = h_test[: min(128, h_test.shape[0])]
    latency = latency_ms(
        lambda: evaluate_generator_gap(
            model,
            w_codebook,
            h_lat,
            n_candidates=args.M,
            n_steps=args.n_steps,
            noisy=args.noisy,
            pt_dbm=args.pt_dbm,
        ),
        repeat=3,
    )
    codebook_metrics = eig_metrics(w_codebook, h_test, noisy=args.noisy, pt_dbm=args.pt_dbm)

    model_path = ensure_parent(args.save_model)
    torch.save(
        {
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "w_codebook": w_codebook.detach().cpu(),
            "settings": vars(args),
            "test_gap": test_gap,
            "latency_ms_per_128_ues": latency,
        },
        model_path,
    )

    result = {
        "settings": vars(args),
        "num_train": int(h_train.shape[0]),
        "num_val": int(h_val.shape[0]),
        "num_test": int(h_test.shape[0]),
        "codebook_path": str(codebook_path),
        "model_path": str(model_path),
        "codebook_metrics": codebook_metrics,
        "codebook_history": codebook_history,
        "cfm_history": cfm_history,
        "test_gap": test_gap,
        "latency_ms_per_128_ues": latency,
    }
    write_json(args.out, result)

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        if codebook_history:
            plot_codebook_loss(codebook_history, plot_dir / "codebook_training_loss.pdf")
        plot_training_curve(cfm_history, "loss", plot_dir / "cfm_training_loss.pdf")
        if any("val_mean_gap_db" in row for row in cfm_history):
            plot_training_curve(cfm_history, "val_mean_gap_db", plot_dir / "cfm_validation_gap.pdf")
        plot_gap_summary(test_gap, plot_dir / "cfm_test_gap_summary.pdf")

    print(f"[done] wrote {args.out}")
    print(f"[done] saved CFM model to {model_path}")


if __name__ == "__main__":
    main()
