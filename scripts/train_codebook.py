import argparse
from pathlib import Path

import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from gen_ssbf.codebook import beam_pattern_db, eig_metrics, train_sim_codebook
from gen_ssbf.data import load_deepmimo_channels, split_train_val_test
from gen_ssbf.plots import plot_beam_patterns, plot_codebook_loss
from gen_ssbf.utils import ensure_parent, resolve_device, seed_all, write_json


def main():
    parser = argparse.ArgumentParser(description="Train a SIM probing codebook.")
    parser.add_argument("--scenarios-dir", required=True)
    parser.add_argument("--scenario", default="o1_28")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--n-antennas", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--corr-weight", type=float, default=0.1)
    parser.add_argument("--coverage-weight", type=float, default=0.0)
    parser.add_argument("--noisy", action="store_true")
    parser.add_argument("--pt-dbm", type=float, default=40.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--max-ues", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=2000)
    parser.add_argument("--save-codebook", default="results/sim_codebook.pt")
    parser.add_argument("--out", default="results/codebook_summary.json")
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
    h_train, _, h_test = split_train_val_test(h, split_seed=args.split_seed, max_test=args.max_test)
    w_codebook, history = train_sim_codebook(
        h_train,
        args.K,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        w_corr=args.corr_weight,
        w_cov=args.coverage_weight,
        noisy=args.noisy,
        pt_dbm=args.pt_dbm,
        seed=args.seed,
    )
    metrics = eig_metrics(w_codebook, h_test, noisy=args.noisy, pt_dbm=args.pt_dbm)

    codebook_path = ensure_parent(args.save_codebook)
    torch.save(
        {
            "w_codebook": w_codebook.detach().cpu(),
            "settings": vars(args),
            "metrics": metrics,
        },
        codebook_path,
    )

    result = {
        "settings": vars(args),
        "num_train": int(h_train.shape[0]),
        "num_test": int(h_test.shape[0]),
        "metrics": metrics,
        "history": history,
        "codebook_path": str(codebook_path),
    }
    write_json(args.out, result)

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        plot_codebook_loss(history, plot_dir / "codebook_training_loss.pdf")
        plot_beam_patterns(beam_pattern_db(w_codebook), plot_dir / "sim_beam_patterns.pdf")

    print(f"[done] wrote {args.out}")
    print(f"[done] saved codebook to {codebook_path}")


if __name__ == "__main__":
    main()
