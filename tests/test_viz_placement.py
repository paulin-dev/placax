import matplotlib

matplotlib.use("Agg")

import numpy as np

from placax_viz.placement import plot_placement, save_placement_image


def test_plot_placement_skips_unplaced_macros() -> None:
    positions = np.array([[1, 1], [-1, -1], [5, 5]])
    sizes = np.array([[2, 3], [4, 4], [1, 1]])

    ax = plot_placement(positions, sizes, grid_x=8)

    assert len(ax.patches) == 3  # the die-boundary rectangle, plus the two placed macros
    assert ax.get_title() == ""  # no title unless explicitly requested
    assert ax.get_xticks().size == 0 and ax.get_yticks().size == 0  # no axis clutter


def test_save_placement_image_writes_a_file(tmp_path) -> None:
    positions = np.array([[0, 0]])
    sizes = np.array([[2, 2]])
    save_path = tmp_path / "placement.png"

    save_placement_image(positions, sizes, grid_x=4, grid_y=4, save_path=save_path)

    assert save_path.exists()
