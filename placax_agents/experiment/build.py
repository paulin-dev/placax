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
from dataclasses import dataclass, replace
from typing import Any, Callable

from placax.log import Log  # must precede jax imports
from placax.netlist.order import alphabetical_order
from placax_agents.agents.base import Agent
from placax_agents.agents.baselines import GreedyWiremaskAgent, RandomSearchAgent
from placax_agents.agents.genetic import GeneticAgent
from placax_agents.agents.local_search import LocalSearchAgent
from placax_agents.agents.ppo import PPOAgent
from placax_agents.benchmark import Benchmark
from placax_agents.experiment.config import ExperimentConfig
from placax_agents.experiment.registry import (
    ACTION_SPACES, CELL_PLACERS, INITS, LEGALIZERS, MASKS, OPTIMIZERS, ORDERS, POLICIES,
    REWARDS, STATES, VALIDATORS, VALUE_LOSSES, resolve,
)
from placax_agents.training.algorithm.config import PPOConfig
from placax_agents.training.loops.buffered_train import buffered_train_step
from placax_agents.training.loops.parallel_train import _jitted_parallel_train_step
from placax_agents.training.loops.train import _jitted_train_step

import jax
import jax.numpy as jnp

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
    buffer_size = n_episodes * built.steps_per_episode
    if buffer_size < batch_size:
        Log.warning(
            f"buffer holds {buffer_size} transitions ({n_episodes} episodes x "
            f"{built.steps_per_episode} macros) but batch_size is {batch_size}, so every "
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
            initial_positions=built.initial_positions, n_placed=built.n_placed,
            action_space=built.action_space,
        )

    # Every epoch runs one gradient step per whole minibatch; the short remainder is dropped.
    return step_fn, n_episodes, ppo_epochs * (buffer_size // batch_size)


def _loop_sequential(built):
    """One episode, one gradient update."""

    def step_fn(key, variables, opt_state, running_stats):
        variables, opt_state, running_stats, loss, _final = _jitted_train_step(
            key, variables, opt_state, running_stats, built.optimizer, built.policy.apply,
            built.benchmark.params, built.benchmark.reward_fn, built.benchmark.sizes_array,
            built.benchmark.cell_size, built.state_fn, built.ppo_config, built.extra_illegal_fn,
            built.initial_positions, built.n_placed, built.action_space,
        )
        return variables, opt_state, running_stats, loss

    return step_fn, 1, 1


def _loop_parallel(built, n_envs: int = 8):
    """n_envs episodes vmapped, averaged into one gradient update."""

    def step_fn(key, variables, opt_state, running_stats):
        keys = jax.random.split(key, n_envs)
        variables, opt_state, running_stats, loss, _final = _jitted_parallel_train_step(
            keys, variables, opt_state, running_stats, built.optimizer, built.policy.apply,
            built.benchmark.params, built.benchmark.reward_fn, built.benchmark.sizes_array,
            built.benchmark.cell_size, built.state_fn, built.ppo_config, built.extra_illegal_fn,
            built.initial_positions, built.n_placed, built.action_space,
        )
        return variables, opt_state, running_stats, loss

    return step_fn, n_envs, 1


LOOPS = {"buffered": _loop_buffered, "sequential": _loop_sequential, "parallel": _loop_parallel}
"""loop name -> builder(built, **kwargs) -> (step_fn, episodes_per_iteration,
gradient_steps_per_iteration). The third value is what makes the compute side of a comparison
reportable rather than guessed - see placax_agents/experiment/budget.py."""


@dataclass(frozen=True)
class ResolvedEnvironment:
    """The environment half of a config, live: the pieces EVERY agent must run inside.

    Bundled into one object so an agent builder cannot take some of it and silently skip the
    rest. `as_kwargs()` is what an agent constructor accepts wholesale, which makes "this agent
    honors the whole environment" the default rather than something each builder has to remember.
    """

    state_fn: Any
    extra_illegal_fn: Any
    initial_positions: Any
    n_placed: int
    action_space: Any = None

    def as_kwargs(self) -> dict:
        return {
            "state_fn": self.state_fn,
            "extra_illegal_fn": self.extra_illegal_fn,
            "initial_positions": self.initial_positions,
            "n_placed": self.n_placed,
            "action_space": self.action_space,
        }


@dataclass(frozen=True)
class BuiltExperiment:
    """Everything an ExperimentConfig resolves to, ready to run.

    `policy`, `optimizer`, `ppo_config` and `step_fn` are None for agents that have no such
    thing - a heuristic or a population method - which is why the runner talks to `agent` and
    never to those directly.
    """

    config: ExperimentConfig
    """The RESOLVED config: the one that was handed in, plus the netlist digest that only exists
    once the design has actually been loaded. This is what a manifest should record."""

    benchmark: Benchmark
    state_fn: Any
    extra_illegal_fn: Any
    episodes_per_iteration: int
    initial_positions: Any = None
    n_placed: int = 0
    agent: Agent | None = None
    policy: Any = None
    optimizer: Any = None
    ppo_config: PPOConfig | None = None
    step_fn: StepFn | None = None
    action_space: Any = None
    """What an action is and what it does - `DiscreteGridPlacement` unless the config says
    otherwise. Part of the environment, so every agent in a comparison gets the same one."""

    legalize_fn: Any = None
    """The configured legalizer, or None. Applied by `experiment.export` when a placement is
    written back into the design's own format - not during the episode, since it moves macros the
    constructive kernel has already committed."""

    cell_placer: Any = None
    """Left None by build(): the physical tools are constructed by `experiment.physical`, which
    is where this machine's binary paths are known. Set it to inject a tool directly."""

    validator: Any = None

    @property
    def n_macros(self) -> int:
        return self.benchmark.params.n_macros

    @property
    def steps_per_episode(self) -> int:
        """Macro placements in one episode - fewer than n_macros when the run is warm-started.

        The budget charges what an episode actually costs, so a warm start that hands the agent
        30 of 543 macros buys proportionally more episodes for the same env_step budget rather
        than being billed for placements nobody made.
        """
        if self.action_space is not None:
            return self.action_space.episode_length(self.benchmark.params, self.n_placed)
        return self.n_macros - self.n_placed

    @property
    def env_steps_per_iteration(self) -> int:
        """Macro placements per iteration - the budget's unit of account.

        Declared up front because the runner has to decide whether it can AFFORD an iteration
        before running it. What actually gets charged is the episode count the agent reports
        afterwards, so an agent whose cost varies is still accounted for exactly.
        """
        return self.episodes_per_iteration * self.steps_per_episode

    def init_variables(self, key: jax.Array):
        """Fresh policy variables, shaped from one real observation of this benchmark.

        Only meaningful for agents that have a policy network; kept because several scripts
        rebuild a variables template this way to load a checkpoint into.
        """
        from placax.core import reset

        obs0 = self.state_fn(reset(self.benchmark.params, self.initial_positions),
                             self.benchmark.params, self.benchmark.sizes_array)
        return self.policy.init(key, obs0)


def build_benchmark(config: ExperimentConfig) -> Benchmark:
    """Loads the netlist exactly as the config describes it - order, budget, grid and reward."""
    spec = config.environment.benchmark
    order_fn = resolve(ORDERS, spec.order, spec.path.name, what="order")
    make_reward_fn = resolve(REWARDS, config.environment.reward, spec.grid, what="reward")
    return Benchmark.load(
        spec.path, grid=spec.grid, make_reward_fn=make_reward_fn,
        order_fn=order_fn or alphabetical_order, macro_budget=spec.macro_budget,
        canvas=spec.canvas,
    )


CELL_ONLY = ("discrete_grid",)
"""Spaces whose action is exactly a grid cell - what a `(grid_x, grid_y)` logits map can express."""

CONSTRUCTIVE = ("discrete_grid", "oriented_grid")
"""Spaces that place macros one at a time, in order. The GA decodes to a cell plus an optional
turn, so it drives both; the policy-based and cell-sampling agents drive only the first."""


def _probe_state(action_space, params):
    """A starting state from this space, for asking it what it is like.

    A perturbation space refuses to reset from an empty canvas - it has nothing to move - so it is
    handed a full one. Nothing about the answers below depends on which placement that is; they
    are properties of the space.
    """
    return action_space.reset(
        params,
        None if action_space.constructive else jnp.zeros((params.n_macros, 2), dtype=jnp.int32),
    )


def _names_a_macro(action_space, benchmark) -> bool:
    """Whether this space's steps are ABOUT a particular macro - `target()`'s own answer.

    The question `ActionSpace.target()` was introduced to answer, finally asked by something other
    than a test. A constructive space answers `state.step`; a perturbation space answers UNPLACED,
    because the macro is part of an action nobody has chosen yet.
    """
    from placax.action_space import UNPLACED

    start = _probe_state(action_space, benchmark.params)
    return int(action_space.target(start, benchmark.params)) != UNPLACED


def _require_answerable_constraints(config, benchmark, extra_illegal_fn, action_space) -> None:
    """Refuses an action mask that asks which macro is being placed, under a space with no answer.

    `wiremask_quality` reads `current_macro_size` and a wiremask whose baseline is "the first
    `state.step` macros are down". Under a perturbation space `state.step` is a move counter, so
    both describe an arbitrary macro - and JAX clamps the out-of-range index rather than raising,
    so the mask came back looking perfectly well-formed. That is a legality rule quietly computed
    for the wrong macro, which is worse than no rule.

    Only the mask is checked, not the whole observation: the canvas, positions and placed mask are
    true whatever the space is, and `local_search` reads exactly those. A mask or state that works
    without a current macro says so with `needs_current_macro = False`; absent, it is assumed to
    need one, because everything written before this check did.
    """
    if extra_illegal_fn is None or _names_a_macro(action_space, benchmark):
        return
    if not getattr(extra_illegal_fn, "needs_current_macro", True):
        return
    raise ValueError(
        f"action mask {config.environment.action_mask.name!r} is about the macro being placed "
        f"next, and the {config.environment.action_space.name!r} action space has no such macro: "
        f"every macro is already down and the action names the one to move, so `target()` answers "
        f"UNPLACED and the mask would be computed for whichever macro `state.step` happened to "
        f"land on. Drop the action mask for this space, or use one that reads the whole placement "
        f"and marks itself `needs_current_macro = False`."
    )


def _accepts_orientations(reward_fn) -> bool:
    """Whether `reward_fn` takes the fifth argument `step()` passes under an oriented space.

    Read from the signature rather than discovered by a TypeError halfway through a traced scan,
    where the error names a lambda in a loop and not the reward a config chose. An unintrospectable
    callable (a builtin, a C extension) is given the benefit of the doubt.
    """
    import inspect

    try:
        signature = inspect.signature(reward_fn)
    except (TypeError, ValueError):
        return True
    kinds = [parameter.kind for parameter in signature.parameters.values()]
    if inspect.Parameter.VAR_POSITIONAL in kinds:
        return True
    positional = sum(
        kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for kind in kinds
    )
    return positional >= 5


def _require_orientation_aware_reward(config, benchmark, action_space) -> None:
    """Refuses a reward that cannot see the orientations its action space will choose.

    The GA on `oriented_grid` evolves a turn per macro. If the reward ignores it, the search
    selects on a number that cannot move with the gene - measured before this check existed: one
    placement scored -22.361 both all-north and all-quarter-turned, while its real HPWL moved from
    21.907 to 23.907 and its overlap changed. Better to refuse at build() than to produce a
    plausible number from a placement nobody made.
    """
    starts_oriented = _probe_state(action_space, benchmark.params).orientations is not None
    if not starts_oriented or _accepts_orientations(benchmark.reward_fn):
        return
    raise ValueError(
        f"action space {config.environment.action_space.name!r} chooses a macro ORIENTATION, and "
        f"reward {config.environment.reward.name!r} cannot see one: a turned macro's center and "
        f"pins move, so its reward would describe a placement that was never made. A RewardFn "
        f"used here takes (old_positions, new_positions, old_placed, new_placed, "
        f"orientations=None) - every shipped reward does; see placax/types.py and "
        f"placax/extras/orientation.py for the two transforms that make one orientation-aware."
    )


def policy_action_spaces(policy) -> tuple[str, ...]:
    """Which action spaces this policy's OUTPUT can express, as the policy itself declares.

    A Flax module says so with an `action_spaces` class attribute. Absent, it is assumed to emit
    the `(grid_x, grid_y)` logits map every shipped architecture emits, which is exactly one grid
    cell and therefore `discrete_grid` alone. Read from the policy rather than hard-coded against
    the name "ppo", so a policy with a fourth logits axis for the quarter turn is an architecture
    someone adds - plus a per-turn legality map, which `legal_action_logits` insists on - and not
    an edit to this file.
    """
    return tuple(getattr(policy, "action_spaces", CELL_ONLY))


def _require_space(name: str, env, allowed: tuple[str, ...]) -> None:
    """Refuse a space whose action this agent's own output cannot express.

    A policy emitting `(grid_x, grid_y)` logits has no way to name a macro to move or a turn to
    apply. Feeding its 2-vector to a space expecting three would place macros at coordinates read
    off a macro index - wrong, and silently so.
    """
    space = getattr(env.action_space, "name", "discrete_grid")
    if space not in allowed:
        raise ValueError(
            f"agent {name!r} produces an action this action space cannot use: it drives "
            f"{' or '.join(repr(one) for one in allowed)}, and this config asks for {space!r}. "
            f"Use an agent built for that space (e.g. 'local_search' for 'perturbation'), or "
            f"change EnvironmentSpec.action_space. See placax/action_space.py."
        )


def _build_ppo_agent(config, benchmark, env):
    """PPO's own pieces - policy, optimizer, loop - assembled behind the Agent seam."""
    policy = resolve(POLICIES, config.agent.policy, benchmark, what="policy")
    # What PPO can drive is what its POLICY can emit, so the policy is resolved first and asked.
    _require_space(f"ppo/{config.agent.policy.name}", env, policy_action_spaces(policy))
    ppo_config = build_ppo_config(config.agent.algorithm)

    # The split optimizer's value_coef is derived from the algorithm's, but that happens in
    # `_with_derived_kwargs` on the way into build() rather than here - so the config that gets
    # hashed and written to the manifest carries the value the run actually used.
    optimizer = resolve(OPTIMIZERS, config.agent.optimizer, what="optimizer")

    # The loop closes over everything above, so it is built against a partially-filled
    # BuiltExperiment; nothing it reads is set after this point.
    placeholder = BuiltExperiment(
        config=config, benchmark=benchmark, state_fn=env.state_fn,
        extra_illegal_fn=env.extra_illegal_fn, episodes_per_iteration=0,
        initial_positions=env.initial_positions, n_placed=env.n_placed,
        policy=policy, optimizer=optimizer, ppo_config=ppo_config,
        # The loops read the space from here, so it has to be set before they are built - an
        # episode's length and its transition both come from it.
        action_space=env.action_space,
    )
    if config.agent.loop.name not in LOOPS:
        raise KeyError(f"unknown loop {config.agent.loop.name!r}; registered: {sorted(LOOPS)}")
    step_fn, episodes, gradient_steps = LOOPS[config.agent.loop.name](
        placeholder, **config.agent.loop.kwargs
    )

    agent = PPOAgent(benchmark, policy, optimizer, step_fn, episodes, env.state_fn,
                     env.extra_illegal_fn, env.initial_positions, env.n_placed, gradient_steps,
                     env.action_space)
    return agent, episodes, {"policy": policy, "optimizer": optimizer,
                             "ppo_config": ppo_config, "step_fn": step_fn}


def _build_greedy_wiremask_agent(config, benchmark, env):
    """One deterministic pass, so one episode per iteration and nothing to carry."""
    _require_space("greedy_wiremask", env, CELL_ONLY)
    agent = GreedyWiremaskAgent(benchmark, **env.as_kwargs(), **config.agent.algorithm.kwargs)
    return agent, 1, {}


def _build_random_search_agent(config, benchmark, env):
    """A population of random legal placements per iteration; keeps the best seen."""
    _require_space("random_search", env, CELL_ONLY)
    agent = RandomSearchAgent(benchmark, **env.as_kwargs(), **config.agent.algorithm.kwargs)
    return agent, agent.population, {}


def _build_genetic_agent(config, benchmark, env):
    """A population that breeds - the first non-sequential algorithm family here."""
    _require_space("genetic", env, CONSTRUCTIVE)
    agent = GeneticAgent(benchmark, **env.as_kwargs(), **config.agent.algorithm.kwargs)
    return agent, agent.population, {}


def _build_local_search_agent(config, benchmark, env):
    """One annealing episode per iteration, over the perturbation space it requires."""
    space = getattr(env.action_space, "name", "discrete_grid")
    if space != "perturbation":
        raise ValueError(
            f"agent 'local_search' moves macros that are already placed, which only the "
            f"'perturbation' action space can express - this config asks for {space!r}. Set "
            f"EnvironmentSpec.action_space to Spec('perturbation')."
        )
    agent = LocalSearchAgent(benchmark, **env.as_kwargs(), **config.agent.algorithm.kwargs)
    return agent, 1, {}


AGENTS = {
    "ppo": _build_ppo_agent,
    "local_search": _build_local_search_agent,
    "greedy_wiremask": _build_greedy_wiremask_agent,
    "random_search": _build_random_search_agent,
    "genetic": _build_genetic_agent,
}
"""algorithm name -> builder(config, benchmark, env) -> (agent, episodes_per_iteration, extra
BuiltExperiment fields), where `env` is the ResolvedEnvironment every agent must run inside.
Adding an agent family is one entry.

Every builder takes the SAME `env` and is expected to hand all of it to its agent. That is not a
convention: a builder that quietly drops the action mask (as both baseline builders used to)
produces an agent running under different constraints from the one it is about to be compared
against, while `assert_comparable` still reports the two environments as identical - a false
negative in exactly the check this machinery exists to provide."""


INITIAL_PLACEMENT_KEY = 0
"""The RNG key every warm start is drawn with - fixed, and deliberately not the run's seed.

The initial placement is part of the ENVIRONMENT: `environment_hash` asserts that two runs being
compared started from the same one, and TILOS's finding is that the starting placement can matter
as much as the method refining it. Drawing it from `config.seed` made that assertion false the
moment anyone registered a stochastic warm start - three seeds of one experiment would have
started from three different placements while every hash reported the warm start as identical.

So the warm start is a pure function of the environment spec. Variation that is supposed to
matter goes in the component's own kwargs, where it is hashed and written down like everything
else: `Spec("my_warm_start", {"sample_seed": 3})` is a different environment and says so, which
is exactly the property a bare `config.seed` could not give it.

Both shipped warm starts ignore the key entirely - `empty` returns None and
`greedy_wiremask_prefix` is deterministic - so no run this project has produced is affected."""


def resolve_initial_placement(config: ExperimentConfig, benchmark: Benchmark, key,
                              action_space=None):
    """The run's warm start, resolved once: (initial_positions, how many macros it placed).

    Once, not per episode, for two reasons. The number of macros left to place has to be a
    static shape for the rollout scan; and "the initial placement" is a property of the
    environment being compared, so resampling it inside the run would make two runs of the same
    config differ on an axis the config claims to pin down.

    `key` is `INITIAL_PLACEMENT_KEY` for the same reason - see there.
    """
    init_fn = resolve(INITS, config.environment.initial_placement, benchmark,
                      what="initial placement")
    initial_positions = init_fn(key)
    if initial_positions is None:
        return None, 0
    n_placed = int((initial_positions[:, 0] >= 0).sum())
    # "Everything is already placed" is only a mistake for a CONSTRUCTIVE space, where it leaves
    # the agent nothing to append. A perturbation space requires exactly that - there is nothing
    # to move otherwise - so the question to ask is whether the episode has any actions left in
    # it, which is the action space's own answer.
    remaining = (
        action_space.episode_length(benchmark.params, n_placed) if action_space is not None
        else benchmark.params.n_macros - n_placed
    )
    if remaining <= 0:
        raise ValueError(
            f"initial placement {config.environment.initial_placement.name!r} pre-placed every "
            f"one of the {benchmark.params.n_macros} macros, leaving the agent nothing to do "
            f"under the {getattr(action_space, 'name', 'discrete_grid')!r} action space"
        )
    return initial_positions, n_placed


def build_physical(config: ExperimentConfig, machine: dict | None = None):
    """(cell_placer, validator) for this config, or (None, None) for a proxy-only run.

    Which tools ran is named by the config, so it is hashed and lands in the manifest. WHERE those
    tools live is not: `machine` carries dreamplace_root, use_docker, gpu, the openroad binary -
    everything that differs between two labs running the same experiment, and that would wreck a
    comparison if it were folded into the environment hash.

    Called by `evaluate_physical`, not by `build()`. A proxy-only experiment must not have to know
    where DREAMPlace is installed just to construct itself.
    """
    machine = machine or {}
    physical = config.environment.physical
    cell_placer = (
        resolve(CELL_PLACERS, _with_machine(physical.cell_placer, machine), what="cell placer")
        if physical.cell_placer is not None else None
    )
    validator = (
        resolve(VALIDATORS, _with_machine(physical.validator, machine), what="validator")
        if physical.validator is not None else None
    )
    return cell_placer, validator


def _with_machine(spec, machine: dict):
    """`spec` with this machine's kwargs merged in. The config's own values always win."""
    from dataclasses import replace as _replace

    return _replace(spec, kwargs={**machine, **spec.kwargs})


def _with_derived_kwargs(config: ExperimentConfig) -> ExperimentConfig:
    """The config with any kwarg one component derives from another written in explicitly.

    Exactly one such coupling exists today: `maskplace_split`'s critic clip threshold is scaled
    by the value_coef the PPO loss actually applies, so that clipping `value_coef * g_v` triggers
    on the same raw ||g_v|| the reference clips. Deriving it here rather than inside the agent
    builder is what keeps the record honest - the config that gets HASHED and written to the
    manifest then carries the value the run really used, instead of recording the optimizer's own
    default while the run quietly uses PPO's.
    """
    agent = config.agent
    optimizer = agent.optimizer
    if (
        optimizer is None
        or optimizer.name != "maskplace_split"
        or agent.algorithm.name != "ppo"
        or "value_coef" in optimizer.kwargs
    ):
        return config
    value_coef = build_ppo_config(agent.algorithm).value_coef
    return replace(config, agent=replace(agent, optimizer=replace(
        optimizer, kwargs={**optimizer.kwargs, "value_coef": value_coef}
    )))


def build(config: ExperimentConfig, benchmark: Benchmark | None = None) -> BuiltExperiment:
    """Resolves a config into live objects. Pass `benchmark` to reuse an already-loaded netlist."""
    import jax.random

    benchmark = benchmark if benchmark is not None else build_benchmark(config)
    # Fold in the netlist's content digest now that the design has actually been parsed: from
    # here on, this config identifies the design by what it contains rather than where it lives.
    config = config.with_benchmark(
        config.environment.benchmark.with_digest(benchmark.netlist_digest)
    )
    # ...and any kwarg one component derives from another, for the same reason: what is recorded
    # has to be what ran.
    config = _with_derived_kwargs(config)

    # The environment half is built the same way whatever the agent is - which is the point:
    # swapping the agent must not be able to change the benchmark, reward, observation, mask or
    # warm start.
    state_fn = resolve(STATES, config.environment.state, benchmark, what="state")
    extra_illegal_fn = (
        resolve(MASKS, config.environment.action_mask, benchmark, what="action mask")
        if config.environment.action_mask is not None else None
    )
    action_space = resolve(
        ACTION_SPACES, config.environment.action_space, benchmark, what="action space"
    )
    # The two things a swapped action space can ask of the rest of the environment, checked here
    # rather than discovered as a wrong number later: can the reward see the orientations this
    # space chooses, and can the legality rules answer "which macro is this step about".
    _require_orientation_aware_reward(config, benchmark, action_space)
    _require_answerable_constraints(config, benchmark, extra_illegal_fn, action_space)
    # Drawn with a FIXED key, not the run's seed: the warm start is part of the environment, and
    # `environment_hash` claims two runs being compared started from the same one. See
    # INITIAL_PLACEMENT_KEY. Validated against the action space, which decides what "nothing left
    # to do" means.
    initial_positions, n_placed = resolve_initial_placement(
        config, benchmark, jax.random.PRNGKey(INITIAL_PLACEMENT_KEY), action_space
    )
    env = ResolvedEnvironment(state_fn, extra_illegal_fn, initial_positions, n_placed, action_space)
    legalize_fn = (
        resolve(LEGALIZERS, config.environment.legalization, benchmark, what="legalizer")
        if config.environment.legalization is not None else None
    )

    algorithm = config.agent.algorithm
    if algorithm.name not in AGENTS:
        raise KeyError(
            f"unknown algorithm {algorithm.name!r}; registered: {', '.join(sorted(AGENTS))}. "
            f"SHAC needs a continuous action space, a smoothed wirelength and a differentiable "
            f"density term; the second of those now exists (the `smoothed` reward) and the other "
            f"two do not - see docs/Action_Space_Decision.md."
        )
    agent, episodes, extras = AGENTS[algorithm.name](config, benchmark, env)

    warm_start = f", warm start {n_placed} macros" if n_placed else ""
    # The space's own episode length, not an assumed macro count: a perturbation episode is a move
    # budget, so this line used to report a number the budget never charged.
    steps_per_episode = action_space.episode_length(benchmark.params, n_placed)
    Log.info(
        f"  {len(benchmark.macro_sizes)} macros, {len(benchmark.nets)} nets, "
        f"cell_size={benchmark.cell_size:.2f}, agent={agent.name}, {episodes} episodes/iteration "
        f"({episodes * steps_per_episode:,} env steps/iteration){warm_start}"
    )
    return BuiltExperiment(
        config=config, benchmark=benchmark, state_fn=state_fn,
        extra_illegal_fn=extra_illegal_fn, episodes_per_iteration=episodes,
        initial_positions=initial_positions, n_placed=n_placed, agent=agent,
        action_space=action_space, legalize_fn=legalize_fn, **extras,
    )
