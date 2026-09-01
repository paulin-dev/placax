"""Static images of a macro placement, rendering each macro as a colored rectangle."""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def plot_placement(positions, sizes, grid_x: int, grid_y: int | None = None, ax=None, title: str | None = None):
    """Draws each placed macro as a rectangle on a grid_x x grid_y canvas, skipping unplaced ones; returns the Axes."""
    positions = np.asarray(positions)
    sizes = np.asarray(sizes)
    grid_y = grid_x if grid_y is None else grid_y

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.set_xlim(0, grid_x)
    ax.set_ylim(0, grid_y)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    cmap = plt.get_cmap("viridis")

    n_macros = len(positions)
    for i, ((x, y), (w, h)) in enumerate(zip(positions, sizes)):
        if x < 0:
            continue
        ax.add_patch(Rectangle((x, y), w, h, facecolor=cmap(i / max(n_macros - 1, 1)), edgecolor="black", linewidth=0.3))

    if title is not None:
        ax.set_title(title)
    return ax


def save_placement_image(
    positions, sizes, grid_x: int, grid_y: int | None, save_path: pathlib.Path | str, title: str | None = None
) -> None:
    """plot_placement, saved to save_path as a standalone figure with no surrounding margin."""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1 if title is None else 0.92)
    plot_placement(positions, sizes, grid_x, grid_y, ax=ax, title=title)
    fig.savefig(save_path, dpi=150, pad_inches=0)
    plt.close(fig)
