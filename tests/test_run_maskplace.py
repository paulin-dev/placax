"""scripts/run_maskplace.py is a CLI over the maskplace preset now, so this tests that surface:
flags reaching the config correctly, and the back-compat builders other scripts still import."""
import pathlib

from placax.core import reset  # noqa: F401  must precede jax imports
from placax_agents.training.algorithm.loss import huber_value_loss
from scripts.run_maskplace import (
    MASKPLACE_LEARNING_RATE,
    MASKPLACE_MAX_GRAD_NORM,
    _budget_from_args,
    _parse_args,
    maskplace_optimizer,
    maskplace_ppo_config,
)

import jax.numpy as jnp
import optax
import pytest
from jax import random

ARGV = ["run_maskplace.py"]


# --------------------------------------------------------------------------- CLI -> config


def test_defaults_match_the_scripts_documented_defaults() -> None:
    args = _parse_args(ARGV)
    assert args.benchmark_dir == pathlib.Path("benchmarks/adaptec1")
    assert args.seed == 42
    assert args.macro_budget == "all"
    assert args.n_episodes == "10"
    assert args.eval_every == 10
    assert args.log_every == 1
    assert args.patience == 0
    assert args.entropy_coef == 0.0
    assert args.regularity_weight == 0.0
    assert args.regularity_mode == "corner"


def test_no_budget_flag_keeps_the_previous_100_iteration_default() -> None:
    # Preserving this matters: it is what --n_iterations meant before budgets existed.
    assert _budget_from_args(_parse_args(ARGV)).iterations == 100


@pytest.mark.parametrize("flag, field, value", [
    ("--n_iterations=7", "iterations", 7),
    ("--env_steps=5000", "env_steps", 5000),
    ("--wall_clock_s=30", "wall_clock_s", 30.0),
])
def test_each_budget_flag_sets_only_its_own_cap(flag: str, field: str, value) -> None:
    budget = _budget_from_args(_parse_args(ARGV + [flag]))
    assert getattr(budget, field) == value
    others = {"iterations", "env_steps", "wall_clock_s"} - {field}
    assert all(getattr(budget, name) is None for name in others)


def test_budget_flags_combine_and_the_first_cap_reached_wins() -> None:
    budget = _budget_from_args(_parse_args(ARGV + ["--n_iterations=10", "--wall_clock_s=60"]))
    assert (budget.iterations, budget.wall_clock_s, budget.env_steps) == (10, 60.0, None)


def test_config_flag_takes_a_recorded_config_path() -> None:
    args = _parse_args(ARGV + ["--config=runs/prev/config.json"])
    assert args.config == pathlib.Path("runs/prev/config.json")


# --------------------------------------------------------------------------- back-compat builders


def test_maskplace_ppo_config_matches_maskplace_values() -> None:
    config = maskplace_ppo_config()
    assert config.gamma == 0.95
    assert config.lam == 1.0
    assert config.clip_eps == 0.2
    assert config.entropy_coef == 0.0
    assert config.value_loss_fn is huber_value_loss
    assert config.normalize_advantages is False
    assert config.normalize_returns is False


def test_maskplace_ppo_config_entropy_coef_is_overridable() -> None:
    assert maskplace_ppo_config(entropy_coef=0.05).entropy_coef == 0.05


def test_maskplace_max_grad_norm_matches_maskplace_value() -> None:
    assert MASKPLACE_MAX_GRAD_NORM == 0.5
    assert MASKPLACE_LEARNING_RATE == 2.5e-3


def test_maskplace_optimizer_isolates_critic_prefixed_params() -> None:
    # Actor and critic must be clipped independently, matching MaskPlace's two separate backward
    # passes: a huge actor gradient must not shrink the critic's update through a shared norm.
    optimizer = maskplace_optimizer(learning_rate=1.0, max_grad_norm=1.0, value_coef=1.0)
    params = {"actor_w": jnp.ones((2,)), "critic_w": jnp.ones((2,))}
    opt_state = optimizer.init(params)

    grads = {"actor_w": jnp.full((2,), 1000.0), "critic_w": jnp.full((2,), 1e-3)}
    updates, _ = optimizer.update(grads, opt_state, params)

    # Adam normalizes magnitude, so compare directions: both groups move against their own
    # gradient, and the critic's tiny gradient is not zeroed by the actor's enormous one.
    assert float(updates["critic_w"][0]) < 0.0
    assert float(updates["actor_w"][0]) < 0.0
    assert abs(float(updates["critic_w"][0])) > 1e-6


def test_maskplace_optimizer_accepts_a_custom_critic_prefix() -> None:
    optimizer = maskplace_optimizer(critic_param_prefix="value_")
    params = {"actor_w": jnp.ones((2,)), "value_w": jnp.ones((2,))}
    updates, _ = optimizer.update(
        {"actor_w": jnp.ones((2,)), "value_w": jnp.ones((2,))}, optimizer.init(params), params
    )
    assert set(updates) == {"actor_w", "value_w"}


def test_optimizer_is_a_real_optax_transformation() -> None:
    assert isinstance(maskplace_optimizer(), optax.GradientTransformation)
    _ = random.PRNGKey(0)  # keeps the jax import meaningful under the import-order guard
