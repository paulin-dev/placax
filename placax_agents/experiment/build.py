"""Turns an ExperimentConfig into the live objects a run needs, and into ONE uniform step function.

The uniform step function is the load-bearing part. The three training loops in this project have
three different signatures and three different notions of an "iteration" - one episode and one
gradient step for the sequential loop, n_envs episodes for the parallel one, n_episodes plus
several minibatch epochs for the buffered one. That is exactly why `--n_iterations` meant three
different amounts of compute and why no two runs could be given a matching budget.

`build()` hides all three behind `StepFn`, and pairs each with the number of episodes it actually
collects per call, so the caller can charge the budget in env steps rather than in iterations.
A future non-gradient agent (ACO, GA) becomes another entry in LOOPS with the same signature; it
needs no changes anywhere else.
"""
from dataclasses import dataclass
from typing import Any, Callable

from placax.log import Log  # must precede jax imports
from placax.netlist.order import alphabetical_order
from placax_agents.benchmark import Benchmark
from placax_agents.experiment.config import ExperimentConfig
from placax_agents.experiment.registry import (
    MASKS, OPTIMIZERS, ORDERS, POLICIES, REWARDS, STATES, VALUE_LOSSES, resolve,
)
from placax_agents.training.algorithm.config import PPOConfig
from placax_agents.training.loops.buffered_train import buffered_train_step
from placax_agents.training.loops.parallel_train import _jitted_parallel_train_step
from placax_agents.training.loops.train import _jitted_train_step

import jax

StepFn = Callable[..., tuple[Any, Any, Any, Any]]
"""(key, variables, opt_state, running_stats) -> (variables, opt_state, running_stats, loss).

Every agent family reduces to this. Nothing outside the loop implementation needs to know
whether the update came from one episode or a buffer of ten, or whether it used a gradient.
"""


def build_ppo_config(spec) -> PPOConfig:
    """PPOConfig from an algorithm Spec, resolving its value-loss name to the real function."""
    if spec.name != "ppo":
        raise KeyError(
            f"unknown algorithm {spec.name!r}; only 'ppo' is implemented. SHAC, ACO and GA are "
            f"the research plan, not shipped - see docs/JAX_Placement_Environment_Spec.md §12."
        )
    kwargs = dict(spec.kwargs)
    value_loss = kwargs.pop("value_loss", "mse")
    if value_loss not in VALUE_LOSSES:
        raise KeyError(f"unknown value_loss {value_loss!r}; registered: {sorted(VALUE_LOSSES)}")
    return PPOConfig(value_loss_fn=VALUE_LOSSES[value_loss], **kwargs)


def _loop_buffered(built, n_episodes: int = 10, ppo_epochs: int = 10, batch_size: int = 64):
    """MaskPlace's own procedure: collect a buffer of episodes, then several reshuffled
    minibatch epochs over it."""
    # buffered_train_step drops the short remainder (matching MaskPlace's BatchSampler(drop_last
    # =True)), so a buffer smaller than one batch yields zero minibatches and the update is
    # silently skipped - the run trains nothing while reporting loss=0.0 every iteration. Easy to
    # hit by lowering --macro_budget or --n_episodes for a quick probe, and invisible without this.
    buffer_size = n_episodes * built.benchmark.params.n_macros
    if buffer_size < batch_size:
        Log.warning(
            f"buffer holds {buffer_size} transitions ({n_episodes} episodes x "
            f"{built.benchmark.params.n_macros} macros) but batch_size is {batch_size}, so every "
            f"update is SKIPPED and the policy never changes. Lower batch_size below "
            f"{buffer_size}, or raise n_episodes/macro_budget."
        )

    def step_fn(key, variables, opt_state, running_stats):
        return buffered_train_step(
            key, variables, opt_state, running_stats, built.optimizer, built.policy.apply,
            built.benchmark.params, built.benchmark.reward_fn, built.benchmark.sizes_array,
            built.benchmark.cell_size, n_episodes, ppo_epochs=ppo_epochs, batch_size=batch_size,
            state_fn=built.state_fn, ppo_config=built.ppo_config,
            extra_illegal_fn=built.extra_illegal_fn,
        )

    return step_fn, n_episodes


def _loop_sequential(built):
    """One episode, one gradient update."""

    def step_fn(key, variables, opt_state, running_stats):
        variables, opt_state, running_stats, loss, _final = _jitted_train_step(
            key, variables, opt_state, running_stats, built.optimizer, built.policy.apply,
            built.benchmark.params, built.benchmark.reward_fn, built.benchmark.sizes_array,
            built.benchmark.cell_size, built.state_fn, built.ppo_config, built.extra_illegal_fn,
        )
        return variables, opt_state, running_stats, loss

    return step_fn, 1


def _loop_parallel(built, n_envs: int = 8):
    """n_envs episodes vmapped, averaged into one gradient update."""

    def step_fn(key, variables, opt_state, running_stats):
        keys = jax.random.split(key, n_envs)
        variables, opt_state, running_stats, loss, _final = _jitted_parallel_train_step(
            keys, variables, opt_state, running_stats, built.optimizer, built.policy.apply,
            built.benchmark.params, built.benchmark.reward_fn, built.benchmark.sizes_array,
            built.benchmark.cell_size, built.state_fn, built.ppo_config, built.extra_illegal_fn,
        )
        return variables, opt_state, running_stats, loss

    return step_fn, n_envs


LOOPS = {"buffered": _loop_buffered, "sequential": _loop_sequential, "parallel": _loop_parallel}


@dataclass(frozen=True)
class BuiltExperiment:
    """Everything an ExperimentConfig resolves to, ready to run."""

    config: ExperimentConfig
    benchmark: Benchmark
    policy: Any
    state_fn: Any
    extra_illegal_fn: Any
    optimizer: Any
    ppo_config: PPOConfig
    step_fn: StepFn
    episodes_per_iteration: int

    @property
    def n_macros(self) -> int:
        return self.benchmark.params.n_macros

    @property
    def env_steps_per_iteration(self) -> int:
        """Macro placements per training iteration - the budget's unit of account."""
        return self.episodes_per_iteration * self.n_macros

    def init_variables(self, key: jax.Array):
        """Fresh policy variables, shaped from one real observation of this benchmark."""
        from placax.core import reset

        obs0 = self.state_fn(reset(self.benchmark.params), self.benchmark.params,
                             self.benchmark.sizes_array)
        return self.policy.init(key, obs0)


def build_benchmark(config: ExperimentConfig) -> Benchmark:
    """Loads the netlist exactly as the config describes it - order, budget, grid and reward."""
    spec = config.environment.benchmark
    order_fn = resolve(ORDERS, spec.order, spec.path.name, what="order")
    make_reward_fn = resolve(REWARDS, config.environment.reward, spec.grid, what="reward")
    return Benchmark.load(
        spec.path, grid=spec.grid, make_reward_fn=make_reward_fn,
        order_fn=order_fn or alphabetical_order, macro_budget=spec.macro_budget,
    )


def build(config: ExperimentConfig, benchmark: Benchmark | None = None) -> BuiltExperiment:
    """Resolves a config into live objects. Pass `benchmark` to reuse an already-loaded netlist."""
    benchmark = benchmark if benchmark is not None else build_benchmark(config)

    state_fn = resolve(STATES, config.environment.state, benchmark, what="state")
    extra_illegal_fn = (
        resolve(MASKS, config.environment.action_mask, benchmark, what="action mask")
        if config.environment.action_mask is not None else None
    )
    policy = resolve(POLICIES, config.agent.policy, benchmark, what="policy")
    ppo_config = build_ppo_config(config.agent.algorithm)

    # The split optimizer needs the value_coef the loss actually applies, so its clip threshold
    # lands on the same raw gradient norm the reference clips. Supplying it here keeps that
    # coupling in one place instead of asking every config to restate the number consistently.
    optimizer_spec = config.agent.optimizer
    if optimizer_spec.name == "maskplace_split" and "value_coef" not in optimizer_spec.kwargs:
        optimizer_spec = type(optimizer_spec)(
            name=optimizer_spec.name,
            kwargs={**optimizer_spec.kwargs, "value_coef": ppo_config.value_coef},
        )
    optimizer = resolve(OPTIMIZERS, optimizer_spec, what="optimizer")

    # Built last: the loop closes over everything above.
    placeholder = BuiltExperiment(
        config=config, benchmark=benchmark, policy=policy, state_fn=state_fn,
        extra_illegal_fn=extra_illegal_fn, optimizer=optimizer, ppo_config=ppo_config,
        step_fn=None, episodes_per_iteration=0,
    )
    if config.agent.loop.name not in LOOPS:
        raise KeyError(f"unknown loop {config.agent.loop.name!r}; registered: {sorted(LOOPS)}")
    step_fn, episodes = LOOPS[config.agent.loop.name](placeholder, **config.agent.loop.kwargs)

    Log.info(
        f"  {len(benchmark.macro_sizes)} macros, {len(benchmark.nets)} nets, "
        f"cell_size={benchmark.cell_size:.2f}, {episodes} episodes/iteration "
        f"({episodes * benchmark.params.n_macros:,} env steps/iteration)"
    )
    return BuiltExperiment(
        config=config, benchmark=benchmark, policy=policy, state_fn=state_fn,
        extra_illegal_fn=extra_illegal_fn, optimizer=optimizer, ppo_config=ppo_config,
        step_fn=step_fn, episodes_per_iteration=episodes,
    )
