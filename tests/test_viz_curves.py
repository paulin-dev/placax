import pathlib

import matplotlib

matplotlib.use("Agg")

from placax_viz.curves import load_training_log, plot_training_curves


def test_load_training_log_parses_jsonl(tmp_path: pathlib.Path) -> None:
    log_path = tmp_path / "training_log.jsonl"
    log_path.write_text('{"iteration": 1, "loss": 0.5, "real_hpwl": null}\n{"iteration": 2, "loss": 0.3, "real_hpwl": 100.0}\n')

    log = load_training_log(log_path)

    assert log == [
        {"iteration": 1, "loss": 0.5, "real_hpwl": None},
        {"iteration": 2, "loss": 0.3, "real_hpwl": 100.0},
    ]


def test_plot_training_curves_accepts_a_path_and_saves_a_file(tmp_path: pathlib.Path) -> None:
    log_path = tmp_path / "training_log.jsonl"
    log_path.write_text('{"iteration": 1, "loss": 0.5, "real_hpwl": null}\n{"iteration": 2, "loss": 0.3, "real_hpwl": 100.0}\n')
    save_path = tmp_path / "curves.png"

    fig = plot_training_curves(log_path, save_path=save_path)

    assert save_path.exists()
    assert len(fig.axes) == 2


def test_plot_training_curves_handles_no_hpwl_points(tmp_path: pathlib.Path) -> None:
    # No eval iterations logged yet - shouldn't crash on the empty real_hpwl series.
    log = [{"iteration": 1, "loss": 0.5, "real_hpwl": None}]
    fig = plot_training_curves(log)
    assert len(fig.axes) == 2
