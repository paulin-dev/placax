"""Loss and real-HPWL curves from a training log (see placax_agents.ops.resumable_train)."""
import json
import pathlib

import matplotlib.pyplot as plt


def load_training_log(log_path: pathlib.Path) -> list[dict]:
    """Reads a training_log.jsonl file (one {iteration, loss, real_hpwl} object per line)."""
    with open(log_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_training_curves(log: list[dict] | pathlib.Path | str, save_path: pathlib.Path | str | None = None):
    """Plots PPO loss alongside real HPWL side by side; returns the Figure and optionally saves it."""
    if isinstance(log, (str, pathlib.Path)):
        log = load_training_log(pathlib.Path(log))

    iterations = [entry["iteration"] for entry in log]
    losses = [entry["loss"] for entry in log]
    hpwl_points = [(entry["iteration"], entry["real_hpwl"]) for entry in log if entry["real_hpwl"] is not None]

    fig, (ax_loss, ax_hpwl) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(iterations, losses, color="#3b6fd6", linewidth=1)
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("PPO loss")
    ax_loss.set_title("Training loss")

    if hpwl_points:
        hpwl_iters, hpwl_values = zip(*hpwl_points)
        ax_hpwl.plot(hpwl_iters, hpwl_values, color="#d6763b", marker="o", markersize=3, linewidth=1)
    ax_hpwl.set_xlabel("iteration")
    ax_hpwl.set_ylabel("real HPWL")
    ax_hpwl.set_title("Placement quality (greedy eval)")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig
