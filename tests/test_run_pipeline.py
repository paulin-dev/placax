import pathlib

from placax import _device  # noqa: F401  must precede jax imports
from placax_agents.ops.checkpoint import save_checkpoint
from placax_tools.cell_placer import CellPlacer
from placax_tools.dreamplace.cell_placer import DREAMPlaceCellPlacer
from scripts.presets import PRESETS
from scripts.run_pipeline import _build_cell_placer, _parse_args, _resolve_checkpoint

import jax.numpy as jnp


def test_parse_args_defaults_to_all_macros_no_dreamplace_and_maskplace_preset() -> None:
    (
        benchmark_dir, preset, checkpoint, macro_budget, output_dir, dreamplace_root, use_docker, gpu,
        target_density, python_executable, dreamplace_extra_config, viz_resolution,
        nets_sample_fraction, nets_seed,
    ) = _parse_args(["x", "--benchmark_dir=benchmarks/adaptec1"])
    assert preset == "maskplace"  # backward-compatible default: this pipeline used to only support this
    assert macro_budget is None  # "all" is the production default
    assert dreamplace_root is None and use_docker is False
    assert dreamplace_extra_config == {}
    assert viz_resolution == 1024
    assert nets_sample_fraction == 1.0
    assert nets_seed == 0


def test_parse_args_accepts_any_registered_preset() -> None:
    for preset_name in PRESETS:
        (_, preset, *_rest) = _parse_args(["x", f"--preset={preset_name}"])
        assert preset == preset_name


def test_parse_args_macro_budget_integer_overrides_all() -> None:
    (_, _, _, macro_budget, *_rest) = _parse_args(["x", "--macro_budget=64"])
    assert macro_budget == 64


def test_parse_args_dreamplace_extra_config_is_parsed_json() -> None:
    args = _parse_args(["x", '--dreamplace_extra_config={"num_bins_x": 256, "random_seed": 7}'])
    dreamplace_extra_config = args[10]
    assert dreamplace_extra_config == {"num_bins_x": 256, "random_seed": 7}


def test_parse_args_nets_sample_fraction_and_seed() -> None:
    args = _parse_args(["x", "--nets_sample_fraction=0.2", "--nets_seed=7"])
    assert args[12] == 0.2
    assert args[13] == 7


def test_resolve_checkpoint_uses_the_given_default_subdir(tmp_path) -> None:
    # Different presets keep their checkpoints under different default subdirs (output_maskplace,
    # output, ...) - _resolve_checkpoint must look under the ONE the caller's preset actually uses,
    # not a hardcoded "output_maskplace".
    checkpoint_path = tmp_path / "output_training" / "best_checkpoint.bin"
    save_checkpoint({"variables": {"params": {}}, "real_hpwl": jnp.array(1.0)}, checkpoint_path)
    path, bare = _resolve_checkpoint(tmp_path, "output_training", None)
    assert path == checkpoint_path
    assert bare is True


def test_build_cell_placer_returns_a_cell_placer() -> None:
    placer = _build_cell_placer(
        pathlib.Path("/opt/dreamplace"), gpu=False, target_density=1.0, python_executable="python",
        use_docker=False, extra_mounts=(),
    )
    assert isinstance(placer, CellPlacer)
    assert isinstance(placer, DREAMPlaceCellPlacer)


def test_build_cell_placer_forwards_extra_config() -> None:
    placer = _build_cell_placer(
        pathlib.Path("/opt/dreamplace"), gpu=False, target_density=1.0, python_executable="python",
        use_docker=False, extra_mounts=(), extra_config={"num_bins_x": 128},
    )
    assert placer.extra_config == {"num_bins_x": 128}
