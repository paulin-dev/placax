"""Completing a Spec's kwargs with its component's own defaults, before anything hashes it.

A config names each component by a registry key plus kwargs, and two people describing the same
experiment do not write the same kwargs: one passes `Spec("hpwl")`, the other spells out
`Spec("hpwl", {"dense": False, "reward_scale": 1.0})`. That is one reward with one set of values,
and before this it produced two different hashes - so `assert_comparable` rejected two identical
setups as incomparable, and a config read back from JSON could never match one built from a
preset. A hash over the *spelling* of a config is not a hash over the config.

So every Spec is completed against its builder's own signature on the way into a hash: a kwarg
nobody wrote down is recorded at the value it will actually run at. That also makes the hash mean
more than it did, because it now covers the defaults a run silently depended on - change a
default in this repo and every config that leaned on it hashes differently, which is right.

Two deliberate exclusions, both because the parameter is not the config's to state:

  * **machinery parameters** (`state_fn`, `initial_positions`, ...) are handed to a component by
    `build()` from the resolved environment. They appear in builder signatures but are never
    written in a config, and they are not JSON at all.
  * **machine parameters** (`dreamplace_root`, `openroad_binary`, `use_docker`, ...). These sit in
    the same builder signatures as real experiment settings, because a tool wrapper needs both to
    be constructed - but where a binary lives differs between two labs running one experiment, and
    folding it into an environment hash would stop those two runs ever comparing as comparable.
    That is what `build_physical`'s machine/config split exists to prevent, and marking the
    parameters is how this file keeps it.

    They used to be avoided by skipping the physical slots WHOLESALE, which cost the slot the
    property every other one has: `Spec("dreamplace")` and `Spec("dreamplace", {"target_density":
    1.0})` are one tool with one setting and hashed differently, so two identical PPA setups
    compared as incomparable - on the one axis where an unattributable number is the whole
    problem.

Everything here is imported lazily. `config.py` must stay importable - and a written config
readable - without pulling in jax, flax, optax and the whole registry behind them.
"""
import functools
import inspect

_JSON_SCALARS = (str, int, float, bool, type(None))
"""What may be recorded in a config, and therefore what may be completed into one. A default of
any other type (a function, a class, an array) belongs to the machinery, not the configuration."""

_MACHINERY_PARAMS = frozenset({
    "state_fn", "extra_illegal_fn", "initial_positions", "n_placed", "benchmark", "params",
    "action_space",
})
"""Parameters `build()` supplies from the ResolvedEnvironment. They carry defaults so a component
can be constructed bare in a test, but a config never states them and a hash must not contain
them."""

MACHINE_PARAMS = frozenset({
    "dreamplace_root", "openroad_binary", "python_executable", "use_docker", "gpu",
    "extra_mounts", "extra_config",
})
"""Where a tool lives and how this host runs it - never part of an experiment's identity.

Two labs running the same experiment on the same design must compare as comparable, so an install
path, a Docker flag or a GPU switch cannot reach a hash. `build_physical` already passes these in
separately as `machine` kwargs; listing them here is what lets everything ELSE in the same
signature - `target_density`, `liberty_path`, `clock_period_ns`, `route` - be completed like any
other component's defaults.

`extra_config` is here for a different reason than the rest: it is a dict, so it is not a JSON
scalar and could never have been completed anyway; naming it keeps the list readable as "the
arguments that are not the experiment"."""


def _defaults_from_signature(builder) -> dict:
    """Every JSON-scalar keyword default `builder` declares, i.e. what a config may leave unsaid.

    A parameter with no default is a leading context argument the registry passes positionally
    (the benchmark, the grid, the partially-built experiment), never a config kwarg - so "has a
    default" is exactly the test for "could have been written in the config".
    """
    try:
        signature = inspect.signature(builder)
    except (TypeError, ValueError):  # a builtin or C-implemented callable has no introspectable signature
        return {}
    defaults = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.default is parameter.empty:
            continue
        if name.startswith("_") or name in _MACHINERY_PARAMS or name in MACHINE_PARAMS:
            continue
        if not isinstance(parameter.default, _JSON_SCALARS):
            continue
        defaults[name] = parameter.default
    return defaults


def _slot_registries() -> dict:
    """slot name -> the registry whose builders carry that slot's defaults."""
    from placax_agents.experiment import registry
    from placax_agents.experiment.build import LOOPS

    return {
        "order": registry.ORDERS,
        "reward": registry.REWARDS,
        "state": registry.STATES,
        "action_mask": registry.MASKS,
        "initial_placement": registry.INITS,
        "action_space": registry.ACTION_SPACES,
        "legalization": registry.LEGALIZERS,
        "policy": registry.POLICIES,
        "optimizer": registry.OPTIMIZERS,
        "loop": LOOPS,
        # The physical slots complete like every other one now; MACHINE_PARAMS above is what
        # keeps this host's install paths out of the result.
        "cell_placer": registry.CELL_PLACERS,
        "validator": registry.VALIDATORS,
    }


def _algorithm_defaults(name: str) -> dict:
    """The algorithm slot's defaults, which do not live in a registry builder's signature.

    PPO's kwargs are PPOConfig's fields (plus `value_loss`, which `build_ppo_config` pops and
    resolves to a function before PPOConfig ever sees it). A non-learning agent's kwargs go
    straight to its constructor, so that is where its defaults are.
    """
    if name == "ppo":
        import dataclasses

        from placax_agents.training.algorithm.config import PPOConfig

        defaults = {
            field.name: field.default
            for field in dataclasses.fields(PPOConfig)
            # value_loss_fn's default is a function - the config states `value_loss`, a name.
            if isinstance(field.default, _JSON_SCALARS)
        }
        defaults["value_loss"] = "mse"  # build_ppo_config's own default
        return defaults

    from placax_agents.agents import baselines
    from placax_agents.agents.genetic import GeneticAgent
    from placax_agents.agents.local_search import LocalSearchAgent
    from placax_agents.agents.shac import SHACAgent

    agent_class = {
        "greedy_wiremask": baselines.GreedyWiremaskAgent,
        "random_search": baselines.RandomSearchAgent,
        "genetic": GeneticAgent,
        "local_search": LocalSearchAgent,
        # SHAC's kwargs are its constructor's, like every agent here except PPO - it carries its
        # own horizon and discount rather than a PPOConfig, and has no loop to name.
        "shac": SHACAgent,
    }.get(name)
    return _defaults_from_signature(agent_class.__init__) if agent_class is not None else {}


@functools.lru_cache(maxsize=None)
def defaults_for(slot: str, name: str) -> dict:
    """Defaults for the component `name` in `slot`, or {} if there is no such registered thing.

    An unknown name is not an error here: a config may legitimately be hashed, written and read
    without ever being built, and `build()` is where naming a component that does not exist
    produces its own readable error. Refusing to hash would just move that error somewhere less
    useful.

    Cached on (slot, name), which assumes the registries are fixed once imported - they are, being
    module-level dicts populated at import. Code that swaps an entry in at runtime (a test, say)
    must call `defaults_for.cache_clear()`, or it will hash against the entry it replaced.
    """
    if slot == "algorithm":
        return _algorithm_defaults(name)
    builder = _slot_registries().get(slot, {}).get(name)
    return _defaults_from_signature(builder) if builder is not None else {}


def complete_kwargs(slot: str, name: str, kwargs: dict) -> dict:
    """`kwargs` with every unstated default filled in. What the config actually runs at."""
    stated = dict(kwargs)
    return {**defaults_for(slot, name), **stated}
