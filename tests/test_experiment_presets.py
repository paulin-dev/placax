"""Pins the shipped presets to the exact behavior the training scripts had before configs existed.

Both presets were extracted from hardcoded choices inside scripts/run_maskplace.py and
scripts/run_training.py. If that extraction changed any value, every result produced before it
becomes incomparable with every result produced after - which is precisely the failure this whole
mechanism exists to prevent. So these tests assert the reference numbers as LITERALS, taken from
the scripts as they were, rather than by reading them back out of the config being tested.
"""
import pathlib

from placax.extras.rewards import make_hpwl_reward  # noqa: F401  must precede jax imports
from placax.netlist.order import connectivity_order_for
from placax_agents.experiment.build import build, build_benchmark, build_ppo_config
from placax_agents.experiment.presets import maskplace, training
from placax_agents.experiment.registry import ORDERS, _maskplace_connectivity_weights
from placax_agents.policy.architectures.cnn import CNNActorCritic
from placax_agents.training.algorithm.loss import huber_value_loss, mse_value_loss
from placax_agents.training.reward import make_expert_reward

import jax.numpy as jnp
import pytest


def _write_tiny_bookshelf(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "sample.aux").write_text(
        "RowBasedPlacement : sample.nodes sample.nets sample.wts sample.pl sample.scl\n"
    )
    (tmp_path / "sample.nodes").write_text(
        "UCLA nodes 1.0\nNumNodes : 3\nNumTerminals : 3\n"
        "a 4 4 terminal\nb 2 2 terminal\nc 2 4 terminal\n"
    )
    (tmp_path / "sample.nets").write_text(
        "UCLA nets 1.0\nNumNets : 2\nNumPins : 4\n"
        "NetDegree : 2 n0\n\ta I : 0.0 0.0\n\tb O : 0.0 0.0\n"
        "NetDegree : 2 n1\n\tb I : 0.0 0.0\n\tc O : 0.0 0.0\n"
    )
    return tmp_path


# --------------------------------------------------------------------------- maskplace preset


def test_maskplace_ppo_hyperparameters_match_ppo2_py() -> None:
    # PPO2.py's own values: no entropy bonus, and raw unnormalized advantages and returns.
    config = build_ppo_config(maskplace("benchmarks/adaptec1").agent.algorithm)
    assert config.gamma == 0.95
    assert config.lam == 1.0
    assert config.clip_eps == 0.2
    assert config.value_coef == 0.5
    assert config.entropy_coef == 0.0
    assert config.value_loss_fn is huber_value_loss
    assert config.normalize_advantages is False
    assert config.normalize_returns is False


def test_maskplace_entropy_coef_is_overridable_without_touching_anything_else() -> None:
    default = build_ppo_config(maskplace("b").agent.algorithm)
    raised = build_ppo_config(maskplace("b", entropy_coef=0.01).agent.algorithm)
    assert raised.entropy_coef == 0.01
    assert (raised.gamma, raised.lam, raised.normalize_advantages) == (
        default.gamma, default.lam, default.normalize_advantages
    )


def test_maskplace_environment_matches_the_original_script() -> None:
    env = maskplace("benchmarks/adaptec1").environment
    assert env.benchmark.grid == 224                       # MASKPLACE_GRID
    assert env.benchmark.macro_budget is None              # --macro_budget default "all"
    assert env.benchmark.order.name == "connectivity_maskplace"
    assert env.reward.name == "maskplace"
    assert env.reward.kwargs["regularity_weight"] == 0.0   # pure MaskPlace by default
    assert env.state.name == "wiremask"
    assert env.state.kwargs["lookahead"] == 2
    assert env.action_mask.name == "wiremask_quality"
    assert env.action_mask.kwargs["margin"] == 1.0         # --soft_coefficient default


def test_maskplace_agent_matches_the_original_script() -> None:
    agent = maskplace("benchmarks/adaptec1").agent
    assert agent.policy.name == "resnet_coarse_fine"
    assert agent.policy.kwargs["critic_style"] == "step_embedding"
    assert agent.optimizer.name == "maskplace_split"
    assert agent.optimizer.kwargs["learning_rate"] == 2.5e-3   # MASKPLACE_LEARNING_RATE
    assert agent.optimizer.kwargs["max_grad_norm"] == 0.5      # MASKPLACE_MAX_GRAD_NORM
    assert agent.loop.name == "buffered"
    assert agent.loop.kwargs == {"n_episodes": 10, "ppo_epochs": 10, "batch_size": 64}


def test_maskplace_default_seed_is_ppo2_pys_own() -> None:
    assert maskplace("b").seed == 42


@pytest.mark.parametrize("name, expected", [
    ("adaptec1", (1.0, 1000.0)),
    ("ariane133", (30000.0, 1000.0)),   # matched by substring, so "ariane133" picks up "ariane"
    ("bigblue3", (1.0, 100000.0)),
    ("bigblue1", (1.0, 1000.0)),        # only bigblue3 is overridden, not every bigblue
])
def test_maskplace_connectivity_weights_match_the_per_benchmark_overrides(name, expected) -> None:
    assert _maskplace_connectivity_weights(name) == expected


def test_maskplace_order_resolves_to_the_benchmarks_own_weights(tmp_path: pathlib.Path) -> None:
    # The order builder gets the benchmark's directory NAME, so a config pointing at ariane133
    # must resolve to ariane's weights and not the default ones.
    sizes = {"a": (1.0, 1.0), "b": (2.0, 2.0), "c": (1.0, 3.0)}
    nets = [[("a", 0.0, 0.0), ("b", 0.0, 0.0)], [("b", 0.0, 0.0), ("c", 0.0, 0.0)]]
    resolved = ORDERS["connectivity_maskplace"]("ariane133")
    expected = connectivity_order_for(30000.0, 1000.0)
    assert resolved(sizes, nets) == expected(sizes, nets)


def test_maskplace_reward_is_identical_to_the_hand_built_reference(tmp_path: pathlib.Path) -> None:
    # The strongest check available: build the preset's benchmark, then rebuild the reward the way
    # the script used to, and require identical values. Scaling here is subtle (real-unit HPWL
    # delta expressed back in grid units, then divided by 200), so an equal-looking config that
    # scaled differently would silently rescale every reward ever reported.
    benchmark_dir = _write_tiny_bookshelf(tmp_path)
    config = maskplace(benchmark_dir, macro_budget=None)
    benchmark = build_benchmark(config)

    from placax.types import EnvParams

    reference = make_expert_reward(
        benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
        benchmark.sizes_array, benchmark.cell_size,
        EnvParams(grid=224, n_macros=benchmark.params.n_macros),
        dense=True, reward_scale=1.0 / (benchmark.cell_size * 200.0),
        regularity_weight=0.0, regularity_mode="corner",
    )

    n = benchmark.params.n_macros
    old = jnp.full((n, 2), -1)
    new = old.at[0].set(jnp.array([3, 5]))
    old_placed, new_placed = old[:, 0] >= 0, new[:, 0] >= 0
    assert float(benchmark.reward_fn(old, new, old_placed, new_placed)) == float(
        reference(old, new, old_placed, new_placed)
    )


def test_maskplace_regularity_weight_reaches_the_reward(tmp_path: pathlib.Path) -> None:
    # The bug this guards: scripts/presets.py used to drop regularity_weight when rebuilding the
    # maskplace setup, so a policy trained WITH EXPlace's periphery term was reloaded against a
    # benchmark built without it.
    benchmark_dir = _write_tiny_bookshelf(tmp_path)
    plain = build_benchmark(maskplace(benchmark_dir))
    regular = build_benchmark(maskplace(benchmark_dir, regularity_weight=0.5))

    n = plain.params.n_macros
    old = jnp.full((n, 2), -1)
    # A cell far from every edge, where the periphery penalty is largest and so cannot cancel.
    new = old.at[0].set(jnp.array([100, 100]))
    old_placed, new_placed = old[:, 0] >= 0, new[:, 0] >= 0
    assert float(plain.reward_fn(old, new, old_placed, new_placed)) != float(
        regular.reward_fn(old, new, old_placed, new_placed)
    )


# --------------------------------------------------------------------------- training preset


def test_training_preset_matches_the_original_scripts_defaults() -> None:
    config = training("benchmarks/adaptec1")
    env, agent = config.environment, config.agent
    # Benchmark.load()'s own defaults, which run_training.py relied on implicitly.
    assert env.benchmark.grid == 64
    assert env.benchmark.macro_budget is None
    assert env.benchmark.order.name == "alphabetical"
    assert env.reward.name == "hpwl"
    assert env.reward.kwargs["dense"] is False      # sparse/terminal, make_scaled_hpwl_reward's default
    assert env.state.name == "canvas"
    assert env.action_mask is None
    assert agent.policy.name == "cnn"
    assert agent.optimizer.kwargs["learning_rate"] == 3e-4
    assert agent.loop.name == "sequential"          # n_envs=1


def test_training_preset_ppo_defaults_are_the_textbook_ones() -> None:
    config = build_ppo_config(training("b").agent.algorithm)
    assert (config.gamma, config.lam, config.clip_eps) == (0.99, 0.95, 0.2)
    assert config.entropy_coef == 0.01
    assert config.value_loss_fn is mse_value_loss
    assert config.normalize_advantages is True
    assert config.normalize_returns is True


def test_training_preset_switches_to_the_parallel_loop_above_one_env() -> None:
    assert training("b", n_envs=1).agent.loop.name == "sequential"
    parallel = training("b", n_envs=8).agent.loop
    assert parallel.name == "parallel"
    assert parallel.kwargs == {"n_envs": 8}


def test_training_preset_seed_is_the_previously_hardcoded_prngkey(tmp_path) -> None:
    # run_training.py used random.PRNGKey(0) with no flag; keeping 0 as the default means results
    # from before this refactor remain reproducible under it.
    assert training("b").seed == 0


# --------------------------------------------------------------------------- building


def test_build_resolves_the_training_preset_end_to_end(tmp_path: pathlib.Path) -> None:
    benchmark_dir = _write_tiny_bookshelf(tmp_path)
    built = build(training(benchmark_dir))
    assert isinstance(built.policy, CNNActorCritic)
    assert built.extra_illegal_fn is None
    assert built.episodes_per_iteration == 1
    assert built.env_steps_per_iteration == built.n_macros


def test_build_reports_env_steps_per_iteration_from_the_loop_shape(tmp_path: pathlib.Path) -> None:
    # The number the budget is charged in. A buffered loop collecting 10 episodes does ten times
    # the work of a sequential one per "iteration", which is exactly why iterations were never a
    # comparable unit.
    benchmark_dir = _write_tiny_bookshelf(tmp_path)
    sequential = build(training(benchmark_dir, n_envs=1))
    parallel = build(training(benchmark_dir, n_envs=4))
    assert parallel.env_steps_per_iteration == 4 * sequential.env_steps_per_iteration


def test_build_rejects_an_unregistered_component(tmp_path: pathlib.Path) -> None:
    from placax_agents.experiment.config import Spec

    benchmark_dir = _write_tiny_bookshelf(tmp_path)
    config = training(benchmark_dir)
    broken = type(config)(
        name=config.name, seed=config.seed,
        environment=type(config.environment)(
            benchmark=config.environment.benchmark, reward=config.environment.reward,
            state=Spec("no_such_state"), budget=config.environment.budget,
        ),
        agent=config.agent,
    )
    with pytest.raises(KeyError, match="unknown state"):
        build(broken)


def test_build_rejects_an_unimplemented_algorithm() -> None:
    from placax_agents.experiment.config import Spec

    # SHAC/ACO/GA are the research plan, not shipped - failing loudly beats silently running PPO.
    with pytest.raises(KeyError, match="only 'ppo' is implemented"):
        build_ppo_config(Spec("shac"))


def test_buffered_loop_warns_when_the_buffer_cannot_fill_one_batch(tmp_path, caplog) -> None:
    # Silent no-op updates: buffered_train_step drops the short remainder, so a buffer smaller
    # than batch_size skips every update while still reporting a loss of 0.0 each iteration.
    # Found by running the CLI with a small --macro_budget, where it looked like training.
    import logging

    benchmark_dir = _write_tiny_bookshelf(tmp_path)   # 3 macros
    config = maskplace(benchmark_dir, n_episodes=1)   # 3 transitions vs batch_size 64
    with caplog.at_level(logging.WARNING):
        build(config)
    assert "every update is SKIPPED" in caplog.text


def test_buffered_loop_stays_quiet_when_the_buffer_is_large_enough(tmp_path, caplog) -> None:
    import logging

    benchmark_dir = _write_tiny_bookshelf(tmp_path)
    config = maskplace(benchmark_dir, n_episodes=1)
    smaller_batches = type(config)(
        name=config.name, seed=config.seed, environment=config.environment,
        agent=type(config.agent)(
            policy=config.agent.policy, algorithm=config.agent.algorithm,
            optimizer=config.agent.optimizer,
            loop=type(config.agent.loop)("buffered", {"n_episodes": 1, "batch_size": 2}),
        ),
    )
    with caplog.at_level(logging.WARNING):
        build(smaller_batches)
    assert "SKIPPED" not in caplog.text
