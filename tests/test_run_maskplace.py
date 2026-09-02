import pathlib

from placax.core import reset  # noqa: F401  must precede jax imports
from placax.extras.rewards import make_hpwl_reward
from placax.types import EnvParams
from placax_agents.benchmark import Benchmark
from placax_agents.policy.architectures.cnn import CNNActorCritic
from placax_agents.policy.observation import observation
from placax_agents.training.algorithm.loss import huber_value_loss
from scripts.run_maskplace import (
    MASKPLACE_LEARNING_RATE,
    MASKPLACE_MAX_GRAD_NORM,
    _train_and_eval_loop,
    maskplace_optimizer,
    maskplace_ppo_config,
)

import jax.numpy as jnp
import optax
from jax import random


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


def _toy_benchmark() -> Benchmark:
    params = EnvParams(grid=8, n_macros=4)
    sizes_array = jnp.array([[2.0, 2.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]], dtype=jnp.float32)
    padded_pin_idx = jnp.array([[0, 1]])
    padded_pin_offset = jnp.zeros((1, 2, 2), dtype=jnp.float32)
    valid_mask = jnp.array([[True, True]])
    reward_fn = make_hpwl_reward(padded_pin_idx, padded_pin_offset, valid_mask)
    return Benchmark(
        macro_sizes={}, nets=[], params=params, sizes_array=sizes_array, cell_size=1.0,
        reward_fn=reward_fn, padded_pin_idx=padded_pin_idx, padded_pin_offset=padded_pin_offset,
        valid_mask=valid_mask, name_to_idx={},
    )


def test_resumed_run_continues_placement_image_iteration_count(tmp_path: pathlib.Path) -> None:
    # Regression: resuming must count placement-image iterations from where the checkpoint left
    # off (current_iteration = start_iteration + i + 1), not restart the loop's own local counter
    # from 1 - which would both mislabel snapshot filenames and desync the --eval_every cadence.
    benchmark = _toy_benchmark()
    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = observation(reset(benchmark.params), benchmark.params, benchmark.sizes_array)
    variables = policy.init(init_key, obs0)
    optimizer = optax.adam(1e-3)
    ppo_config = maskplace_ppo_config()

    checkpoint_path = tmp_path / "checkpoint.bin"
    images_dir = tmp_path / "placements"

    _train_and_eval_loop(
        key, variables, policy, benchmark, optimizer, ppo_config, observation, None,
        checkpoint_path, n_iterations=5, n_episodes=2,
        log_every=1, eval_every=5, log_path=None, placement_images_dir=images_dir,
    )
    assert sorted(p.name for p in images_dir.glob("*.png")) == ["5.png"]

    # Resume to iteration 15 (a fresh key - resume must read state from checkpoint_path, not
    # depend on continuing the same key/variables objects, matching a real second CLI invocation).
    _train_and_eval_loop(
        random.PRNGKey(1), variables, policy, benchmark, optimizer, ppo_config, observation, None,
        checkpoint_path, n_iterations=15, n_episodes=2,
        log_every=1, eval_every=5, log_path=None, placement_images_dir=images_dir,
    )
    assert sorted(p.name for p in images_dir.glob("*.png")) == ["10.png", "15.png", "5.png"]
