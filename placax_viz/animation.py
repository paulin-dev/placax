"""Builds an animated GIF of a placement rollout, macro by macro (MaskPlace's "Placement" gif)."""
import pathlib

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from placax_viz.placement import plot_placement


def save_placement_gif(
    positions_history: list,
    sizes,
    grid_x: int,
    grid_y: int | None,
    save_path: pathlib.Path | str,
    fps: int = 4,
) -> None:
    """positions_history: one (n_macros, 2) grid-unit positions array per frame."""
    fig, ax = plt.subplots(figsize=(5, 5))

    def draw(frame_idx: int) -> None:
        ax.clear()
        plot_placement(
            positions_history[frame_idx], sizes, grid_x, grid_y, ax=ax,
            title=f"step {frame_idx}/{len(positions_history) - 1}",
        )

    anim = FuncAnimation(fig, draw, frames=len(positions_history), interval=1000 / fps)
    anim.save(str(save_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
