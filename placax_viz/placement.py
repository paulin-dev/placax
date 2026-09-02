"""Static images of a macro placement, rendering each macro as a plain rectangle (matching MaskPlace's own look)."""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from placax.extras.mst import prim_mst_edges


def plot_placement(
    positions,
    sizes,
    grid_x: int,
    grid_y: int | None = None,
    ax=None,
    title: str | None = None,
    macro_edgewidth: float = 0.2,
):
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
        ax.add_patch(Rectangle((x, y), w, h, facecolor="tab:blue", edgecolor="black", linewidth=macro_edgewidth))

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


def plot_net_connections(
    ax,
    positions,
    sizes_array,
    padded_pin_idx,
    padded_pin_offset,
    valid_mask,
    cell_size: float,
    color: str = "black",
    linewidth: float = 0.6,
    line_alpha: float = 0.3,
    marker_size: float = 3.0,
    marker_alpha: float = 0.5,
    sample_fraction: float = 1.0,
    seed: int = 0,
):
    """Draws each net's Manhattan MST edges (via placax.extras.mst.prim_mst_edges) as thin, semi-transparent
    lines with an opaque dot at every pin, in the same grid-unit coordinate system plot_placement's
    Rectangles use - the same MST wirelength proxy already used elsewhere in this codebase, not a new
    topology heuristic. Drawn at a high zorder so lines/dots stay visible over the (opaque) macro
    rectangles, not hidden underneath them. Line and marker opacity are independent so pins stay legible
    even with line_alpha turned down on a dense netlist.

    sample_fraction < 1.0 randomly keeps only that fraction of nets (same convention MaskPlace's own
    Figure 1 uses on its full, unfiltered netlist - "For clarity, we only show 1% wires" - for a less
    cluttered view on a denser netlist than our macro-only one; deterministic given the same seed."""
    positions = np.asarray(positions)
    sizes_array = np.asarray(sizes_array)
    pin_idx = np.asarray(padded_pin_idx)
    pin_offset = np.asarray(padded_pin_offset)
    valid = np.asarray(valid_mask)

    # Real-unit macro centers, one lookup shared by every net below.
    real_centers = positions * cell_size + sizes_array / 2.0

    n_nets = pin_idx.shape[0]
    keep = np.ones(n_nets, dtype=bool)
    if sample_fraction < 1.0:
        keep = np.random.default_rng(seed).random(n_nets) < sample_fraction

    for net_idx in range(n_nets):
        if not keep[net_idx]:
            continue
        net_valid = valid[net_idx]
        macro_ids = pin_idx[net_idx][net_valid]
        offsets = pin_offset[net_idx][net_valid]
        if len(macro_ids) < 2 or np.any(positions[macro_ids, 0] < 0):
            continue  # nothing to connect, or a macro this net touches was never placed
        # Real-unit pin location -> back to grid units, matching the Rectangles already on this Axes.
        pin_points = [tuple((real_centers[m] + off) / cell_size) for m, off in zip(macro_ids, offsets)]
        for i, j in prim_mst_edges(pin_points):
            (x1, y1), (x2, y2) = pin_points[i], pin_points[j]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=line_alpha, zorder=2)
        xs, ys = zip(*pin_points)
        ax.scatter(xs, ys, s=marker_size**2, color=color, alpha=marker_alpha, zorder=3, linewidths=0)
    return ax


def save_placement_with_nets_image(
    positions,
    sizes_array,
    grid_x: int,
    grid_y: int | None,
    padded_pin_idx,
    padded_pin_offset,
    valid_mask,
    cell_size: float,
    save_path: pathlib.Path | str,
    title: str | None = None,
    sample_fraction: float = 1.0,
    seed: int = 0,
) -> None:
    """save_placement_image, with each net's MST connections drawn on top of the macro rectangles.
    sizes_array is REAL-unit macro sizes (matching plot_net_connections' own real-unit math) - converted to
    grid units internally for plot_placement's Rectangles, so callers pass the same sizes_array everywhere
    instead of having to convert it themselves (a previous version of this function took pre-converted grid
    sizes here, silently drawing wildly oversized macros when real-unit sizes were passed instead).
    sample_fraction/seed are forwarded to plot_net_connections (see its docstring - MaskPlace's own "only
    show 1% wires" convention, for a less cluttered view on a denser netlist)."""
    from placax_agents.policy.scale import to_grid_units

    grid_sizes = to_grid_units(sizes_array, cell_size)
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1 if title is None else 0.92)
    plot_placement(positions, grid_sizes, grid_x, grid_y, ax=ax, title=title)
    plot_net_connections(
        ax, positions, sizes_array, padded_pin_idx, padded_pin_offset, valid_mask, cell_size,
        sample_fraction=sample_fraction, seed=seed,
    )
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_full_placement_image(
    macro_positions,
    macro_sizes,
    cell_positions,
    cell_sizes,
    die_width: float,
    die_height: float | None,
    save_path: pathlib.Path | str,
    resolution: int = 1024,
    title: str | None = None,
) -> None:
    """Renders every cell (rasterized into a density grid, so this stays fast for 100k+ cells) plus every
    macro (crisp rectangles on top) - the natural post-cell-placement visualization. Unlike plot_placement,
    positions/sizes here are REAL-unit coordinates: a cell placer's output isn't tied to placax's RL grid."""
    macro_positions = np.asarray(macro_positions, dtype=np.float64)
    macro_sizes = np.asarray(macro_sizes, dtype=np.float64)
    cell_positions = np.asarray(cell_positions, dtype=np.float64)
    cell_sizes = np.asarray(cell_sizes, dtype=np.float64)
    die_height = die_width if die_height is None else die_height

    # 1. Bin every cell's center into a coarse occupancy grid - vectorized, so cell count doesn't matter.
    density = np.zeros((resolution, resolution), dtype=np.float32)
    if len(cell_positions) > 0:
        cell_centers = cell_positions + cell_sizes / 2.0
        bin_x = np.clip((cell_centers[:, 0] / die_width * resolution).astype(np.int64), 0, resolution - 1)
        bin_y = np.clip((cell_centers[:, 1] / die_height * resolution).astype(np.int64), 0, resolution - 1)
        np.add.at(density, (bin_y, bin_x), 1.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1 if title is None else 0.92)
    ax.set_xlim(0, die_width)
    ax.set_ylim(0, die_height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 2. Cell density underneath (log-scaled so a few crowded bins don't wash out the rest), macros on top.
    ax.imshow(
        np.log1p(density), origin="lower", extent=(0, die_width, 0, die_height),
        cmap="Greys", aspect="auto", zorder=0,
    )
    ax.add_patch(Rectangle((0, 0), die_width, die_height, facecolor="none", edgecolor="black", linewidth=1.5))
    for (x, y), (w, h) in zip(macro_positions, macro_sizes):
        ax.add_patch(Rectangle((x, y), w, h, facecolor="tab:blue", edgecolor="black", linewidth=0.2, zorder=1))

    if title is not None:
        ax.set_title(title)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
