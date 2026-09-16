import pathlib
import pytest

from placax import _device  # noqa: F401  must precede jax imports
from placax_agents.ops.checkpoint import save_checkpoint
from scripts.presets import PRESETS
from scripts.run_pipeline import _parse_args, _resolve_checkpoint

import jax.numpy as jnp


def test_parse_args_defaults_to_all_macros_no_dreamplace_and_maskplace_preset() -> None:
    # Read by NAME, not by position. This used to be a fifteen-value tuple unpacked positionally,
    # which broke every caller the moment a flag was added - and one was.
    args = _parse_args(["x", "--benchmark_dir=benchmarks/adaptec1"])
    assert args.preset == "maskplace"  # backward-compatible default: all this pipeline once had
    assert args.macro_budget is None  # "all" is the production default
    assert args.dreamplace_root is None and args.use_docker is False
    assert args.dreamplace_extra_config == {}
    assert args.viz_resolution == 1024
    assert args.nets_sample_fraction == 1.0
    assert args.nets_seed == 0
    assert args.config is None  # the preset-name path stays the default, with a warning at run time


def test_parse_args_carries_a_config_path_when_one_is_given() -> None:
    # --config is what lets this pipeline rebuild the environment a checkpoint was trained in,
    # rather than whatever a preset name resolves to today.
    args = _parse_args(["x", "--config=runs/adaptec1/manifest.json"])
    assert args.config == pathlib.Path("runs/adaptec1/manifest.json")


def test_parse_args_accepts_any_registered_preset() -> None:
    for preset_name in PRESETS:
        assert _parse_args(["x", f"--preset={preset_name}"]).preset == preset_name


def test_parse_args_macro_budget_integer_overrides_all() -> None:
    assert _parse_args(["x", "--macro_budget=64"]).macro_budget == 64


def test_parse_args_dreamplace_extra_config_is_parsed_json() -> None:
    args = _parse_args(["x", '--dreamplace_extra_config={"num_bins_x": 256, "random_seed": 7}'])
    assert args.dreamplace_extra_config == {"num_bins_x": 256, "random_seed": 7}


def test_parse_args_nets_sample_fraction_and_seed() -> None:
    args = _parse_args(["x", "--nets_sample_fraction=0.2", "--nets_seed=7"])
    assert args.nets_sample_fraction == 0.2
    assert args.nets_seed == 7


@pytest.fixture
def runs(tmp_path, monkeypatch) -> pathlib.Path:
    from placax_agents.experiment import presets

    monkeypatch.setattr(presets, "RUNS_DIR", tmp_path / "runs")
    return tmp_path / "runs"


def test_resolve_checkpoint_uses_the_presets_own_run_directory(runs) -> None:
    # Different presets keep their checkpoints under different run names (adaptec1-maskplace,
    # adaptec1-training) - _resolve_checkpoint must look under the ONE the caller's preset uses.
    checkpoint_path = runs / "adaptec1-training" / "best_checkpoint.bin"
    save_checkpoint({"variables": {"params": {}}, "real_hpwl": jnp.array(1.0)}, checkpoint_path)
    path, bare = _resolve_checkpoint(pathlib.Path("benchmarks/adaptec1"), "training", None)
    assert path == checkpoint_path
    assert bare is True


def test_resolve_checkpoint_finds_a_seeded_run_directory(runs) -> None:
    # Runs write to runs/<benchmark>-<preset>/seed<N>, one directory per run (see
    # presets.default_output_dir), so the default lookup has to see that layout as well as a flat one.
    run_dir = runs / "adaptec1-training" / "seed3"
    (run_dir).mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}")
    checkpoint_path = run_dir / "best_checkpoint.bin"
    save_checkpoint({"variables": {"params": {}}, "real_hpwl": jnp.array(1.0)}, checkpoint_path)
    path, _bare = _resolve_checkpoint(pathlib.Path("benchmarks/adaptec1"), "training", None)
    assert path == checkpoint_path


def test_pipeline_output_goes_beside_the_run_that_trained_the_checkpoint(runs) -> None:
    from scripts.run_pipeline import _default_output_dir

    checkpoint = runs / "adaptec1-maskplace" / "seed0" / "best_checkpoint.bin"
    bench = pathlib.Path("benchmarks/adaptec1")
    assert _default_output_dir(checkpoint, "maskplace", bench) == checkpoint.parent / "pipeline"
    # A checkpoint kept anywhere else never pulls generated output in beside it.
    elsewhere = pathlib.Path("benchmarks/adaptec1/old/best_checkpoint.bin")
    assert _default_output_dir(elsewhere, "maskplace", bench) == runs / "adaptec1-maskplace" / "pipeline"


def test_the_tools_this_pipeline_runs_are_written_into_its_config() -> None:
    from placax_agents.experiment.presets import maskplace
    from scripts.run_pipeline import _with_physical_stack

    config = _with_physical_stack(maskplace("benchmarks/adaptec1"), "openroad:route=global",
                                  target_density=0.9, place_cells=True)
    physical = config.environment.physical
    assert physical.cell_placer.name == "dreamplace"
    assert physical.cell_placer.kwargs == {"target_density": 0.9}
    assert physical.validator.name == "openroad" and physical.validator.kwargs == {"route": "global"}
    # Nothing placed, nothing named.
    bare = _with_physical_stack(maskplace("benchmarks/adaptec1"), None, 1.0, place_cells=False)
    assert bare.environment.physical.cell_placer is None


def test_the_machine_carries_every_tools_host_settings(tmp_path) -> None:
    from scripts.run_pipeline import _machine

    args = _parse_args(["x", "--use_docker", "--openroad_binary=/o",
                        '--dreamplace_extra_config={"num_bins_x": 128}'])
    machine = _machine(args, tmp_path / "bench", tmp_path / "out")
    assert machine["use_docker"] is True and machine["openroad_binary"] == "/o"
    assert machine["extra_config"] == {"num_bins_x": 128}
    (tmp_path / "bench").mkdir()
    machine = _machine(args, tmp_path / "bench", tmp_path / "out")
    assert set(machine["extra_mounts"]) == {tmp_path / "bench", tmp_path / "out"}
    # Never their common parent: for a benchmark and an output on different trees that is `/`.
    machine = _machine(args, tmp_path / "bench", pathlib.Path("/tmp") / tmp_path.name / "o")
    assert pathlib.Path("/") not in machine["extra_mounts"]
    assert machine["dreamplace_root"] is not None   # --use_docker supplies the default checkout


def test_a_physical_file_replaces_the_stack_and_a_validator_flag_still_wins(tmp_path) -> None:
    import json

    from placax_agents.experiment.presets import maskplace
    from scripts.run_pipeline import _with_physical_stack

    (tmp_path / "physical.json").write_text(json.dumps({
        "cell_placer": {"name": "openroad", "kwargs": {"density": 0.3}},
        "validator": {"name": "openroad", "kwargs": {"route": "global"}},
    }))
    config = _with_physical_stack(maskplace("benchmarks/adaptec1"), "openroad:route=detailed",
                                  1.0, place_cells=False, physical_path=tmp_path / "physical.json")
    physical = config.environment.physical
    assert physical.cell_placer.name == "openroad"          # not replaced by DREAMPlace
    assert physical.validator.kwargs == {"route": "detailed"}
