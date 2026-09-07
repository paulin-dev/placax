"""Experiment configuration: one serializable description of a run, and one loop that honors it.

The public surface most callers need:

    from placax_agents.experiment import ExperimentConfig, Budget, build, run_experiment, presets

    config = presets.maskplace("benchmarks/adaptec1", budget=Budget(env_steps=5_000_000))
    variables, log = run_experiment(config, output_dir=pathlib.Path("runs/adaptec1-maskplace"))

Two runs may only be compared if `assert_comparable(a, b)` passes - that is, if they share an
environment hash. See config.py for why that split exists.
"""
from placax_agents.experiment.budget import Budget, BudgetTracker, BudgetUse
from placax_agents.experiment.build import BuiltExperiment, build, build_benchmark, build_ppo_config
from placax_agents.experiment.config import (
    AgentSpec, BenchmarkSpec, EnvironmentSpec, ExperimentConfig, Spec, assert_comparable,
)
from placax_agents.experiment.run import run_experiment, write_manifest

__all__ = [
    "AgentSpec", "BenchmarkSpec", "Budget", "BudgetTracker", "BudgetUse", "BuiltExperiment",
    "EnvironmentSpec", "ExperimentConfig", "Spec", "assert_comparable", "build",
    "build_benchmark", "build_ppo_config", "run_experiment", "write_manifest",
]
