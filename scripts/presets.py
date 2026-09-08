"""Rebuilds a training setup from its name, for the scripts that reload a trained policy.

This is now a thin adapter over `placax_agents.experiment`: the presets themselves are configs
(placax_agents/experiment/presets.py) and building one is `experiment.build()`. That matters
because this registry used to re-derive each setup by importing a training script's private
helpers, which let the two drift - and they had: the maskplace entry silently dropped
`regularity_weight`/`regularity_mode`, so a policy trained with EXPlace's periphery term was
reloaded here against a benchmark built without it. Routing both through one config removes the
opportunity for that class of bug rather than fixing this instance of it.

Register a new preset in placax_agents/experiment/presets.py; it becomes available to every
script that consumes this module.
"""
import pathlib

from placax.log import Log
from placax_agents.experiment.build import BuiltExperiment, build
from placax_agents.experiment.config import ExperimentConfig
from placax_agents.experiment.presets import OUTPUT_SUBDIRS, PRESETS as CONFIG_PRESETS, build_preset


def config_for(config_path, preset: str, benchmark_dir, macro_budget=None) -> ExperimentConfig:
    """The config a downstream script should rebuild from: a run's own manifest, or a preset name.

    Every script that reloads a checkpoint needs the environment that checkpoint was trained in -
    the reward, the observation, the action mask, the macro budget, the canvas AND the initial
    placement. A preset name reconstructs a guess at that; a run's `manifest.json` reconstructs the
    thing itself. Hand-matching the two is the error class `ExperimentConfig` removed from
    training, and it survived in the inference scripts long after.

    Falling back is allowed but not silent: a run rebuilt from a preset name gets a warning saying
    what it is assuming.
    """
    if config_path is not None:
        Log.info(f"rebuilding the environment from {config_path}")
        return ExperimentConfig.read(pathlib.Path(config_path))
    overrides = {"macro_budget": macro_budget} if preset == "maskplace" else {}
    Log.warning(
        f"no --config: rebuilding the environment from the preset name {preset!r} alone, which is "
        f"only correct if this checkpoint was trained with exactly that preset. Pass "
        f"--config=<a run's manifest.json> to rebuild the environment it actually used."
    )
    return build_preset(preset, benchmark_dir, **overrides)


def build_setup(config: ExperimentConfig) -> BuiltExperiment:
    """Everything a config resolves to - benchmark, policy, observation, mask, optimizer, loop."""
    return build(config)


def setup_from_preset(name: str, benchmark_dir: pathlib.Path, macro_budget: int | None = None):
    """(benchmark, policy, state_fn, extra_illegal_fn, optimizer) for a named preset.

    The tuple shape is what scripts/run_pipeline.py, scripts/visualize.py and
    scripts/place_once.py consume: they need the pieces to rebuild a policy around a checkpoint,
    not a training loop.
    """
    # `training` places the whole netlist by construction, so a macro budget is meaningless
    # there - passing one would build a config that silently disagrees with its own preset.
    overrides = {"macro_budget": macro_budget} if name == "maskplace" else {}
    built = build(build_preset(name, benchmark_dir, **overrides))
    return built.benchmark, built.policy, built.state_fn, built.extra_illegal_fn, built.optimizer


def _preset_entry(name: str):
    """Adapts one config preset to the (output_subdir, setup_fn) shape the scripts expect."""
    def setup_fn(benchmark_dir: pathlib.Path, macro_budget: int | None):
        return setup_from_preset(name, benchmark_dir, macro_budget)

    return OUTPUT_SUBDIRS[name], setup_fn


PRESETS = {name: _preset_entry(name) for name in CONFIG_PRESETS}
"""preset name -> (default output subdir under benchmark_dir, setup_fn); setup_fn(benchmark_dir,
macro_budget) -> (benchmark, policy, state_fn, extra_illegal_fn, optimizer)."""
