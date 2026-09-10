"""Bringing your own component - the project's founding rule, made checkable.

Section 3 of the spec: *"if a piece of logic could plausibly be done differently by a different
team, it's a parameter, not a hard-coded call."* The registries exist so a config can name a
component in JSON rather than hold a live object, and it is easy to read that as a closed set of
choices. It is not one. These tests pin the two routes a research script actually uses:

  * **register a name** — `register("reward", "mine", fn)`, after which a config selects it like a
    shipped component and round trips through JSON like one; and
  * **hand over a live object** — a `Benchmark` or a whole `BuiltExperiment` built however you
    like, passed straight to `build()` / `run_experiment()`, for a one-off that never needs a
    name.

Nothing in `placax_agents/experiment/registry.py` needs editing for either.
"""
import dataclasses
import functools
import pathlib

import pytest

from placax.core import reset  # noqa: F401  must precede jax imports
from placax_agents.agents.base import UpdateResult
from placax_agents.experiment.budget import Budget
from placax_agents.experiment.build import build, build_benchmark
from placax_agents.experiment.config import AgentSpec, Spec
from placax_agents.experiment.defaults import defaults_for
from placax_agents.experiment.presets import training
from placax_agents.experiment.registry import register, registered, resolve
from placax_agents.experiment.run import run_experiment, score
from placax_agents.training.reward import make_scaled_hpwl_reward

import jax
import jax.numpy as jnp

NODES = ("UCLA nodes 1.0\nNumNodes : 4\nNumTerminals : 4\n"
         "a 4 4 terminal\nb 2 2 terminal\nc 2 4 terminal\nd 4 2 terminal\n")
NETS = ("UCLA nets 1.0\nNumNets : 3\nNumPins : 6\n"
        "NetDegree : 2 n0\n\ta I : 0.0 0.0\n\tb O : 0.0 0.0\n"
        "NetDegree : 2 n1\n\tb I : 0.0 0.0\n\tc O : 0.0 0.0\n"
        "NetDegree : 2 n2\n\tc I : 0.0 0.0\n\td O : 0.0 0.0\n")


def _design(directory: pathlib.Path) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "s.aux").write_text("RowBasedPlacement : s.nodes s.nets s.wts s.pl s.scl\n")
    (directory / "s.nodes").write_text(NODES)
    (directory / "s.nets").write_text(NETS)
    return directory


def _small(config, **environment_overrides):
    return dataclasses.replace(config, environment=dataclasses.replace(
        config.environment,
        benchmark=dataclasses.replace(config.environment.benchmark, grid=12),
        **environment_overrides,
    ))


@pytest.fixture(autouse=True)
def _isolate_registries():
    """Undo anything a test registers, so one test's component cannot leak into another."""
    from placax_agents.experiment import registry as registry_module
    from placax_agents.experiment.build import AGENTS

    before = {name: dict(getattr(registry_module, name))
              for name in ("REWARDS", "STATES", "POLICIES", "ACTION_SPACES")}
    agents_before = dict(AGENTS)
    yield
    for name, contents in before.items():
        getattr(registry_module, name).clear()
        getattr(registry_module, name).update(contents)
    AGENTS.clear()
    AGENTS.update(agents_before)
    defaults_for.cache_clear()


# ------------------------------------------------------------------ a custom reward


def _my_reward(_grid, weight: float = 2.0):
    """An ordinary function, defined outside the library, with its own kwarg."""
    return functools.partial(make_scaled_hpwl_reward, dense=True, reward_scale=weight)


def test_a_reward_defined_outside_the_library_is_selectable_from_a_config(tmp_path) -> None:
    register("reward", "my_reward", _my_reward)
    config = _small(training(_design(tmp_path / "bench"), budget=Budget(iterations=1)),
                    reward=Spec("my_reward", {"weight": 3.0}))
    benchmark = build_benchmark(config)
    assert benchmark.reward_fn is not None
    assert "my_reward" in registered("reward")


def test_a_custom_component_round_trips_through_json_like_a_shipped_one(tmp_path) -> None:
    # A config holds NAMES, not objects, which is exactly why a custom component can survive a
    # results file at all.
    from placax_agents.experiment.config import ExperimentConfig

    register("reward", "my_reward", _my_reward)
    config = _small(training(_design(tmp_path / "bench"), budget=Budget(iterations=1)),
                    reward=Spec("my_reward", {"weight": 3.0}))
    assert ExperimentConfig.from_json(config.to_json()) == config


def test_a_custom_components_own_defaults_reach_the_hash(tmp_path) -> None:
    # The same completion shipped components get: two spellings of one custom reward must hash
    # alike, or `assert_comparable` rejects two identical experiments.
    register("reward", "my_reward", _my_reward)
    directory = _design(tmp_path / "bench")
    spelled = _small(training(directory, budget=Budget(iterations=1)),
                     reward=Spec("my_reward", {"weight": 2.0}))
    bare = _small(training(directory, budget=Budget(iterations=1)), reward=Spec("my_reward"))
    assert defaults_for("reward", "my_reward") == {"weight": 2.0}
    assert spelled.environment_hash() == bare.environment_hash()


def test_register_clears_the_memoized_defaults(tmp_path) -> None:
    """The trap a bare dict write leaves behind.

    `defaults_for` memoizes per (slot, name). A name hashed BEFORE it was registered caches an
    empty default set, and would keep hashing against it forever - so two runs of the same
    experiment could disagree depending on import order. `register` clears the cache; assigning
    into the dict does not.
    """
    config = _small(training(_design(tmp_path / "bench"), budget=Budget(iterations=1)),
                    reward=Spec("my_reward"))
    stale = config.environment_hash()          # hashed before the component exists
    register("reward", "my_reward", _my_reward)
    assert config.environment_hash() != stale, "the cache was not cleared"
    assert defaults_for("reward", "my_reward") == {"weight": 2.0}


# ------------------------------------------------------------------ a custom agent


class _AlwaysCorner:
    """A whole agent from user code: no policy, no optimizer, no gradient, four methods."""

    name = "always_corner"

    def __init__(self, benchmark, corner: int = 0, **_environment):
        self.benchmark = benchmark
        self.corner = corner

    def init(self, key):
        return {}

    def update(self, key, state):
        return state, UpdateResult(episodes=1, loss=None)

    def best_positions(self, state):
        return jnp.full((self.benchmark.params.n_macros, 2), self.corner, dtype=jnp.int32)

    def converged(self, _state) -> bool:
        return True


def test_an_agent_defined_outside_the_library_runs_through_the_shared_loop(tmp_path) -> None:
    """The strongest form of the claim: a foreign agent inherits the whole apparatus.

    Evaluation, checkpointing, budgeting, the manifest and the scoring path all come for free -
    which is the point of the Agent seam, and what stops a new method becoming another
    per-paper one-off environment.
    """
    def build_agent(config, benchmark, env):
        return _AlwaysCorner(benchmark, **config.agent.algorithm.kwargs), 1, {}

    register("algorithm", "always_corner", build_agent)
    directory = _design(tmp_path / "bench")
    config = dataclasses.replace(
        _small(training(directory, budget=Budget(iterations=2))),
        agent=AgentSpec(algorithm=Spec("always_corner", {"corner": 1})),
    )
    output = tmp_path / "run"
    state, log = run_experiment(config, output, eval_every=1)

    assert (output / "manifest.json").exists(), "a foreign agent still leaves a manifest"
    assert log and log[-1]["real_hpwl"] is not None, "and is scored by the runner, not by itself"
    built = build(config)
    assert bool((built.agent.best_positions(state) == 1).all())


def test_a_custom_agent_is_comparable_against_a_shipped_one(tmp_path) -> None:
    from placax_agents.experiment.config import assert_comparable

    def build_agent(config, benchmark, env):
        return _AlwaysCorner(benchmark, **config.agent.algorithm.kwargs), 1, {}

    register("algorithm", "always_corner", build_agent)
    directory = _design(tmp_path / "bench")
    reference = _small(training(directory, budget=Budget(iterations=1)))
    mine = dataclasses.replace(reference, agent=AgentSpec(algorithm=Spec("always_corner")))
    assert_comparable(reference, mine)


# ------------------------------------------------------------------ the object route


def test_a_prebuilt_benchmark_can_be_handed_straight_to_build(tmp_path) -> None:
    """The route for a one-off that never needs a name: build the object yourself.

    `Benchmark.load` takes `make_reward_fn` directly, so a reward that will only ever be used
    once needs no registry entry at all.
    """
    from placax_agents.benchmark import Benchmark

    directory = _design(tmp_path / "bench")
    config = _small(training(directory, budget=Budget(iterations=1)))
    benchmark = Benchmark.load(
        directory, grid=12,
        make_reward_fn=functools.partial(make_scaled_hpwl_reward, dense=True, reward_scale=5.0),
    )
    built = build(config, benchmark=benchmark)
    assert built.benchmark is benchmark
    positions = built.agent.best_positions(built.agent.init(jax.random.PRNGKey(0)))
    assert score(built.benchmark, positions, built.n_placed)["real_hpwl"] > 0


def test_a_built_experiment_can_have_pieces_swapped_before_it_runs(tmp_path) -> None:
    # BuiltExperiment is a frozen dataclass, so `dataclasses.replace` is the supported way to
    # inject a live object - which is how the physical tests substitute stand-in tools.
    config = _small(training(_design(tmp_path / "bench"), budget=Budget(iterations=1)))
    built = build(config)
    swapped = dataclasses.replace(built, agent=_AlwaysCorner(built.benchmark, corner=2))
    assert swapped.agent.name == "always_corner"
    assert swapped.benchmark is built.benchmark


# ------------------------------------------------------------------ the guard rails


def test_registering_into_an_unknown_slot_is_refused() -> None:
    with pytest.raises(KeyError, match="unknown slot"):
        register("vibes", "x", lambda: None)


def test_a_builder_must_be_callable() -> None:
    with pytest.raises(TypeError, match="must be callable"):
        register("reward", "x", "not a function")


def test_an_unknown_name_points_at_register_rather_than_at_editing_the_library() -> None:
    # The error message is the only documentation most people will read at the moment they need
    # it, and it used to say "add it to registry.py".
    from placax_agents.experiment.registry import REWARDS

    with pytest.raises(KeyError, match="registries are open"):
        resolve(REWARDS, Spec("nope"), 12, what="reward")


def test_registered_lists_shipped_and_custom_together() -> None:
    register("reward", "my_reward", _my_reward)
    names = registered("reward")
    assert "hpwl" in names and "my_reward" in names


def test_every_slot_a_config_can_name_is_registerable() -> None:
    """A slot missing from SLOTS is a component nobody can extend - the closed-set failure.

    Checked against the config's own fields rather than a hand-kept list, so a new swappable axis
    that forgets to expose itself here fails a test instead of silently being shipped-only.
    """
    from placax_agents.experiment.registry import SLOTS

    environment_slots = {"order", "reward", "state", "action_mask", "action_space",
                         "initial_placement", "legalization", "cell_placer", "validator"}
    agent_slots = {"algorithm", "policy", "optimizer", "loop"}
    assert environment_slots | agent_slots <= set(SLOTS)
    for slot in SLOTS:
        assert registered(slot) is not None
