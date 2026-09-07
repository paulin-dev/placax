"""Experiment configuration: one serializable description of a run, and one loop that honors it.

The public surface most callers need:

    from placax_agents.experiment import ExperimentConfig, Budget, build, run_experiment, presets

    config = presets.maskplace("benchmarks/adaptec1", budget=Budget(env_steps=5_000_000))
    variables, log = run_experiment(config, output_dir=pathlib.Path("runs/adaptec1-maskplace"))

Two runs may only be compared if `assert_comparable(a, b)` passes - that is, if they match at
the level the comparison claims (benchmark / task / environment / full). See config.py for why
that split exists, and why there is more than one level.

`experiment.physical` is imported separately rather than re-exported here: it pulls in the
tool wrappers, and a wirelength-proxy experiment must not need DREAMPlace installed to run.
"""
from placax_agents.experiment.budget import Budget, BudgetTracker, BudgetUse
from placax_agents.experiment.build import (
    BuiltExperiment, ResolvedEnvironment, build, build_benchmark, build_physical, build_ppo_config,
)
from placax_agents.experiment.config import (
    COMPARISON_LEVELS, AgentSpec, BenchmarkSpec, EnvironmentSpec, ExperimentConfig, PhysicalSpec,
    Spec, assert_comparable,
)
from placax_agents.experiment.run import run_experiment, score, score_placement, write_manifest

__all__ = [
    "COMPARISON_LEVELS", "AgentSpec", "BenchmarkSpec", "Budget", "BudgetTracker", "BudgetUse",
    "BuiltExperiment", "EnvironmentSpec", "ExperimentConfig", "PhysicalSpec",
    "ResolvedEnvironment", "Spec", "assert_comparable", "build", "build_benchmark",
    "build_physical", "build_ppo_config", "run_experiment", "score", "score_placement",
    "write_manifest",
]
