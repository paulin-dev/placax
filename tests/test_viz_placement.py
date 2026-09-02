import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from placax_viz.placement import (
    plot_net_connections, plot_placement, save_full_placement_image, save_placement_image,
    save_placement_with_nets_image,
)


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


def test_plot_net_connections_draws_one_line_per_net() -> None:
    positions = np.array([[0, 0], [4, 4]])
    sizes = np.array([[2.0, 2.0], [2.0, 2.0]])
    # One net connecting both macros' centers (zero pin offset each).
    padded_pin_idx = np.array([[0, 1]])
    padded_pin_offset = np.zeros((1, 2, 2))
    valid_mask = np.array([[True, True]])

    _, ax = plt.subplots()
    plot_net_connections(ax, positions, sizes, padded_pin_idx, padded_pin_offset, valid_mask, cell_size=1.0)

    assert len(ax.lines) == 1  # a 2-pin net's MST is a single edge


def test_plot_net_connections_skips_unplaced_macros() -> None:
    positions = np.array([[0, 0], [-1, -1]])  # second macro never placed
    sizes = np.array([[2.0, 2.0], [2.0, 2.0]])
    padded_pin_idx = np.array([[0, 1]])
    padded_pin_offset = np.zeros((1, 2, 2))
    valid_mask = np.array([[True, True]])

    _, ax = plt.subplots()
    plot_net_connections(ax, positions, sizes, padded_pin_idx, padded_pin_offset, valid_mask, cell_size=1.0)

    assert len(ax.lines) == 0


def test_plot_net_connections_draws_a_dot_marker_at_each_pin() -> None:
    # Regression: the reference "what good net rendering looks like" has a visible dot at every pin,
    # not just a bare line. Markers are a separate scatter (its own alpha), not an inline plot() marker.
    positions = np.array([[0, 0], [4, 4]])
    sizes = np.array([[2.0, 2.0], [2.0, 2.0]])
    padded_pin_idx = np.array([[0, 1]])
    padded_pin_offset = np.zeros((1, 2, 2))
    valid_mask = np.array([[True, True]])

    _, ax = plt.subplots()
    plot_net_connections(ax, positions, sizes, padded_pin_idx, padded_pin_offset, valid_mask, cell_size=1.0)

    assert len(ax.collections) == 1  # one scatter call for this net's pins
    assert len(ax.collections[0].get_offsets()) == 2  # both pins marked


def test_plot_net_connections_line_and_marker_alpha_are_independent() -> None:
    positions = np.array([[0, 0], [4, 4]])
    sizes = np.array([[2.0, 2.0], [2.0, 2.0]])
    padded_pin_idx = np.array([[0, 1]])
    padded_pin_offset = np.zeros((1, 2, 2))
    valid_mask = np.array([[True, True]])

    _, ax = plt.subplots()
    plot_net_connections(
        ax, positions, sizes, padded_pin_idx, padded_pin_offset, valid_mask, cell_size=1.0,
        line_alpha=0.1, marker_alpha=0.9,
    )

    assert ax.lines[0].get_alpha() == 0.1
    assert ax.collections[0].get_alpha() == 0.9


def test_plot_net_connections_sample_fraction_is_deterministic_and_reduces_nets() -> None:
    rng_positions = np.arange(20).reshape(10, 2) * 5
    sizes = np.full((10, 2), 2.0)
    # 10 independent 2-pin nets, one per consecutive macro pair.
    padded_pin_idx = np.array([[i, i + 1] for i in range(0, 9, 2)] * 1)
    padded_pin_offset = np.zeros((len(padded_pin_idx), 2, 2))
    valid_mask = np.ones((len(padded_pin_idx), 2), dtype=bool)

    _, ax_full = plt.subplots()
    plot_net_connections(
        ax_full, rng_positions, sizes, padded_pin_idx, padded_pin_offset, valid_mask, cell_size=1.0,
    )
    _, ax_sampled = plt.subplots()
    plot_net_connections(
        ax_sampled, rng_positions, sizes, padded_pin_idx, padded_pin_offset, valid_mask, cell_size=1.0,
        sample_fraction=0.3, seed=42,
    )
    _, ax_sampled_again = plt.subplots()
    plot_net_connections(
        ax_sampled_again, rng_positions, sizes, padded_pin_idx, padded_pin_offset, valid_mask, cell_size=1.0,
        sample_fraction=0.3, seed=42,
    )

    assert len(ax_sampled.lines) < len(ax_full.lines)
    assert len(ax_sampled.lines) == len(ax_sampled_again.lines)  # same seed -> same subsample


def test_save_placement_with_nets_image_scales_macros_to_grid_units(tmp_path, monkeypatch) -> None:
    # Regression: an earlier version passed REAL-unit sizes straight into plot_placement (which expects
    # grid units), drawing macros many times too large and completely covering the net-connection lines.
    # With cell_size=10 a 20x20 real-unit macro must render as a 2x2 grid-unit Rectangle, not 20x20.
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)  # keep the figure open so we can inspect it
    positions = np.array([[0, 0], [4, 4]])
    sizes = np.array([[20.0, 20.0], [20.0, 20.0]])
    padded_pin_idx = np.array([[0, 1]])
    padded_pin_offset = np.zeros((1, 2, 2))
    valid_mask = np.array([[True, True]])
    save_path = tmp_path / "scaled.png"

    save_placement_with_nets_image(
        positions, sizes, 8, 8, padded_pin_idx, padded_pin_offset, valid_mask, cell_size=10.0,
        save_path=save_path,
    )

    macro_patches = [p for p in plt.gcf().axes[0].patches if 0 < p.get_width() < 8]
    assert len(macro_patches) == 2
    for patch in macro_patches:
        assert patch.get_width() == 2.0 and patch.get_height() == 2.0


def test_save_placement_with_nets_image_writes_a_file(tmp_path) -> None:
    positions = np.array([[0, 0], [4, 4]])
    sizes = np.array([[2.0, 2.0], [2.0, 2.0]])
    padded_pin_idx = np.array([[0, 1]])
    padded_pin_offset = np.zeros((1, 2, 2))
    valid_mask = np.array([[True, True]])
    save_path = tmp_path / "placement_with_nets.png"

    save_placement_with_nets_image(
        positions, sizes, 8, 8, padded_pin_idx, padded_pin_offset, valid_mask, cell_size=1.0,
        save_path=save_path,
    )

    assert save_path.exists()


def test_save_full_placement_image_writes_a_file(tmp_path) -> None:
    macro_positions = np.array([[0.0, 0.0]])
    macro_sizes = np.array([[10.0, 10.0]])
    # Many cells, to exercise the vectorized density-binning path.
    rng = np.random.default_rng(0)
    cell_positions = rng.uniform(0, 90, size=(500, 2))
    cell_sizes = np.full((500, 2), 1.0)
    save_path = tmp_path / "full_placement.png"

    save_full_placement_image(
        macro_positions, macro_sizes, cell_positions, cell_sizes, die_width=100.0, die_height=100.0,
        save_path=save_path,
    )

    assert save_path.exists()
