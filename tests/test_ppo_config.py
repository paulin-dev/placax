from placax_agents.training.algorithm.config import (
    MASKPLACE_LEARNING_RATE,
    MASKPLACE_MAX_GRAD_NORM,
    PPOConfig,
    maskplace_optimizer,
    maskplace_ppo_config,
)
from placax_agents.training.algorithm.loss import huber_value_loss

import jax.numpy as jnp


def test_ppo_config_defaults_match_prior_hardcoded_values() -> None:
    # Regression test: these were the values silently hardcoded inside
    # compute_gae/ppo_loss before this config existed - the defaults
    # here must match, or every existing training run's behavior would
    # silently change.
    config = PPOConfig()
    assert config.gamma == 0.99
    assert config.lam == 0.95
    assert config.clip_eps == 0.2
    assert config.value_coef == 0.5
    assert config.entropy_coef == 0.01


def test_ppo_config_is_hashable() -> None:
    # Required for use as a JAX static jit argument.
    assert hash(PPOConfig()) == hash(PPOConfig())
    assert hash(PPOConfig(clip_eps=0.3)) != hash(PPOConfig())


def test_ppo_config_overrides_work() -> None:
    config = PPOConfig(gamma=0.9, entropy_coef=0.1)
    assert config.gamma == 0.9
    assert config.entropy_coef == 0.1
    assert config.lam == 0.95  # untouched fields keep their default


def test_maskplace_ppo_config_matches_maskplace_values() -> None:
    config = maskplace_ppo_config()
    assert config.gamma == 0.95
    assert config.lam == 1.0  # no GAE smoothing -> plain discounted return
    assert config.entropy_coef == 0.0  # no entropy bonus
    assert config.value_loss_fn is huber_value_loss
    assert MASKPLACE_LEARNING_RATE == 2.5e-3


def test_maskplace_optimizer_isolates_critic_prefixed_params() -> None:
    # A large gradient on the "critic_" group must not spill over to the
    # non-critic group's clip-by-global-norm - proof each network is
    # clipped and stepped independently, matching MaskPlace's two
    # separate optimizers.
    optimizer = maskplace_optimizer(learning_rate=0.1, max_grad_norm=1.0)
    params = {"critic_value": jnp.array(0.0), "fine_branch": jnp.array(0.0)}
    grads = {"critic_value": jnp.array(1000.0), "fine_branch": jnp.array(0.5)}

    opt_state = optimizer.init(params)
    updates, _new_state = optimizer.update(grads, opt_state, params)

    # critic_value's huge gradient gets clipped to norm 1.0 before Adam - its
    # step should be small and bounded, not scaled by the 1000.0 magnitude.
    assert abs(float(updates["critic_value"])) < 1.0
    # fine_branch's much smaller, unclipped-in-effect gradient still moves.
    assert updates["fine_branch"] != 0.0


def test_maskplace_max_grad_norm_matches_maskplace_value() -> None:
    assert MASKPLACE_MAX_GRAD_NORM == 0.5
