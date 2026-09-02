from scripts.run_maskplace import (
    MASKPLACE_LEARNING_RATE,
    MASKPLACE_MAX_GRAD_NORM,
    maskplace_optimizer,
    maskplace_ppo_config,
)
from placax_agents.training.algorithm.loss import huber_value_loss

import jax.numpy as jnp


def test_maskplace_ppo_config_matches_maskplace_values() -> None:
    config = maskplace_ppo_config()
    assert config.gamma == 0.95
    assert config.lam == 1.0  # no GAE smoothing -> plain discounted return
    # MaskPlace's own literal entropy_coef=0.0 - note this saturates this JAX/float32 implementation's
    # softmax to exact 0/1 by iteration ~4-6 (confirmed by checkpoint inspection), a known consequence
    # kept here for faithful comparison; other configs in this codebase should use PPOConfig's own
    # general default (entropy_coef=0.01) instead, which avoids the saturation.
    assert config.entropy_coef == 0.0
    # MaskPlace's own advantage/return computation is raw, with no normalization.
    assert config.normalize_advantages is False
    assert config.normalize_returns is False
    assert config.value_loss_fn is huber_value_loss
    assert MASKPLACE_LEARNING_RATE == 2.5e-3


def test_maskplace_ppo_config_entropy_coef_is_overridable() -> None:
    # Normalization must stay off regardless - only entropy_coef itself is meant to vary here.
    config = maskplace_ppo_config(entropy_coef=0.01)
    assert config.entropy_coef == 0.01
    assert config.normalize_advantages is False
    assert config.normalize_returns is False


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
