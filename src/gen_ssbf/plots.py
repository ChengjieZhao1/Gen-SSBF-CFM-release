from pathlib import Path

import matplotlib.pyplot as plt


def _prepare(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_codebook_loss(history, path):
    path = _prepare(path)
    plt.figure(figsize=(5.0, 3.4))
    plt.plot([row["epoch"] for row in history], [row["loss"] for row in history])
    plt.xlabel("Epoch")
    plt.ylabel("SIM objective")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_beam_patterns(pattern, path):
    path = _prepare(path)
    spatial, response_db = pattern
    plt.figure(figsize=(5.6, 3.6))
    for i in range(response_db.shape[1]):
        plt.plot(spatial, response_db[:, i], linewidth=0.7, alpha=0.7)
    plt.xlabel("Normalized spatial frequency")
    plt.ylabel("Normalized response (dB)")
    plt.ylim([-35, 1])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_training_curve(history, metric, path):
    path = _prepare(path)
    x = [row["epoch"] for row in history if metric in row]
    y = [row[metric] for row in history if metric in row]
    plt.figure(figsize=(5.0, 3.4))
    plt.plot(x, y)
    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " "))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_gap_summary(gap_metrics, path):
    path = _prepare(path)
    labels = ["mean", "median", "p10", "p05"]
    values = [
        gap_metrics["mean_gap_db"],
        gap_metrics["median_gap_db"],
        gap_metrics["p10_gap_db"],
        gap_metrics["p05_gap_db"],
    ]
    plt.figure(figsize=(5.0, 3.2))
    plt.bar(labels, values)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("Gap to MRT (dB)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
