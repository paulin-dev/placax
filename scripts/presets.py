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

from placax_agents.experiment.build import BuiltExperiment, build
from placax_agents.experiment.config import ExperimentConfig
from placax_agents.experiment.presets import OUTPUT_SUBDIRS, PRESETS as CONFIG_PRESETS, build_preset


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
