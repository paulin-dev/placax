import matplotlib

matplotlib.use("Agg")

import numpy as np

from placax_viz.masks import plot_observation_channels


def test_plot_observation_channels_one_panel_per_channel() -> None:
    obs = {"canvas": np.zeros((4, 4), dtype=bool)}
    fig = plot_observation_channels(obs)
    assert len(fig.axes) == 1

    obs["wiremask"] = np.zeros((4, 4))
    fig = plot_observation_channels(obs)
    assert len(fig.axes) == 2

    obs["lookahead_wiremasks"] = np.zeros((2, 4, 4))
    fig = plot_observation_channels(obs)
    assert len(fig.axes) == 4  # canvas + wiremask + 2 lookahead slices
