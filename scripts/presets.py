"""Shared presets: given only --benchmark_dir (+ --macro_budget for maskplace), rebuild the exact
benchmark/policy/state_fn/extra_illegal_fn/optimizer setup a specific training script trains with - so
any script that needs to reload a trained policy (scripts/visualize.py, scripts/run_pipeline.py) stays
usable with ANY registered preset, not hardcoded to one training script's own private helpers. Add a new
preset here (and to PRESETS) to make it available to every script that consumes this registry."""
import functools
import pathlib

from placax_agents.benchmark import Benchmark
from placax_agents.policy.architectures.cnn import CNNActorCritic
from placax_agents.policy.observation import observation
from placax_agents.types import ExtraIllegalFn, StateFn

import optax


def training_setup(benchmark_dir: pathlib.Path, _macro_budget: int | None):
    """scripts/run_training.py's own setup: full netlist, plain CNN policy/observation."""
    benchmark = Benchmark.load(benchmark_dir)
    policy = CNNActorCritic()
    # Bare `observation` defaults to cell_size=1.0 - bind the benchmark's real cell_size instead.
    state_fn: StateFn = functools.partial(observation, cell_size=benchmark.cell_size)
    extra_illegal_fn: ExtraIllegalFn | None = None
    optimizer = optax.adam(3e-4)
    return benchmark, policy, state_fn, extra_illegal_fn, optimizer


def maskplace_setup(benchmark_dir: pathlib.Path, macro_budget: int | None):
    """scripts/run_maskplace.py's own setup, imported lazily so other presets skip its extra deps."""
    from scripts.run_maskplace import (
        WIREMASK_MARGIN,
        _build_policy,
        _build_state_fn,
        _load_benchmark,
        maskplace_optimizer,
        maskplace_ppo_config,
    )
    from placax_agents.policy.action import make_wiremask_quality_illegal

    benchmark = _load_benchmark(benchmark_dir, macro_budget)
    policy = _build_policy(benchmark)
    state_fn = _build_state_fn(benchmark)
    extra_illegal_fn = make_wiremask_quality_illegal(margin=WIREMASK_MARGIN, cell_size=benchmark.cell_size)
    optimizer = maskplace_optimizer(value_coef=maskplace_ppo_config().value_coef)
    return benchmark, policy, state_fn, extra_illegal_fn, optimizer


PRESETS = {"training": ("output", training_setup), "maskplace": ("output_maskplace", maskplace_setup)}
"""preset name -> (default output subdir under benchmark_dir, setup_fn); setup_fn(benchmark_dir,
macro_budget) -> (benchmark, policy, state_fn, extra_illegal_fn, optimizer)."""
