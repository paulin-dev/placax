"""Panels of what the policy sees at one step: the placement canvas, plus any per-cell cost
channels (e.g. a wiremask) the observation dict happens to carry, side by side."""
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def plot_observation_channels(obs: dict, save_path: pathlib.Path | str | None = None):
    """Plots obs["canvas"] plus, if present, obs["wiremask"]/obs["lookahead_wiremasks"]
    (see policy.observation.make_wiremask_observation) as heatmap panels. Returns the Figure."""
    panels = [("canvas", np.asarray(obs["canvas"]).T)]
    if "wiremask" in obs:
        panels.append(("wiremask", np.asarray(obs["wiremask"]).T))
    if "lookahead_wiremasks" in obs:
        for i, wiremask in enumerate(np.asarray(obs["lookahead_wiremasks"])):
            panels.append((f"lookahead[{i}]", wiremask.T))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    axes = [axes] if len(panels) == 1 else axes
    for ax, (name, data) in zip(axes, panels):
        ax.imshow(data, origin="lower", cmap="Greys" if name == "canvas" else "inferno")
        ax.set_title(name)
        ax.axis("off")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig
