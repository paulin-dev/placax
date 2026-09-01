"""Static images of a macro placement, rendering each macro as a plain rectangle (matching MaskPlace's own look)."""
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

    # A small margin outside the die boundary so the chip doesn't touch the image edge, matching
    # the reference MaskPlace figures (their autoscale-with-default-margin look), while still
    # keeping the die itself at true full-canvas scale rather than auto-cropping to content.
    margin = 0.03 * max(grid_x, grid_y)
    ax.set_xlim(-margin, grid_x + margin)
    ax.set_ylim(-margin, grid_y + margin)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # The die boundary itself, drawn explicitly so it's always there regardless of axes styling.
    ax.add_patch(Rectangle((0, 0), grid_x, grid_y, facecolor="none", edgecolor="black", linewidth=1.5))

    for (x, y), (w, h) in zip(positions, sizes):
        if x < 0:
            continue
        ax.add_patch(Rectangle((x, y), w, h, facecolor="tab:blue", edgecolor="black", linewidth=1.0))

    if title is not None:
        ax.set_title(title)
    return ax


def save_placement_image(
    positions, sizes, grid_x: int, grid_y: int | None, save_path: pathlib.Path | str, title: str | None = None
) -> None:
    """plot_placement, saved to save_path as a standalone figure with a small margin around the die."""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1 if title is None else 0.92)
    plot_placement(positions, sizes, grid_x, grid_y, ax=ax, title=title)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
