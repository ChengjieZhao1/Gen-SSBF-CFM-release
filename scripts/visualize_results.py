import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from gen_ssbf.plots import plot_codebook_loss, plot_gap_summary, plot_training_curve
from gen_ssbf.utils import read_json


def main():
    parser = argparse.ArgumentParser(description="Create figures from saved training summaries.")
    parser.add_argument("--codebook-json", default=None)
    parser.add_argument("--cfm-json", default=None)
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    written = []

    if args.codebook_json:
        data = read_json(args.codebook_json)
        if data.get("history"):
            path = out_dir / "codebook_training_loss.pdf"
            plot_codebook_loss(data["history"], path)
            written.append(path)

    if args.cfm_json:
        data = read_json(args.cfm_json)
        if data.get("codebook_history"):
            path = out_dir / "codebook_training_loss.pdf"
            plot_codebook_loss(data["codebook_history"], path)
            written.append(path)
        if data.get("cfm_history"):
            path = out_dir / "cfm_training_loss.pdf"
            plot_training_curve(data["cfm_history"], "loss", path)
            written.append(path)
            if any("val_mean_gap_db" in row for row in data["cfm_history"]):
                path = out_dir / "cfm_validation_gap.pdf"
                plot_training_curve(data["cfm_history"], "val_mean_gap_db", path)
                written.append(path)
        if data.get("test_gap"):
            path = out_dir / "cfm_test_gap_summary.pdf"
            plot_gap_summary(data["test_gap"], path)
            written.append(path)

    if not written:
        raise SystemExit("No figures were generated. Pass --codebook-json, --cfm-json, or both.")

    for path in written:
        print(f"[done] wrote {path}")


if __name__ == "__main__":
    main()
