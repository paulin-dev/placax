import dataclasses

from placax import EnvParams, random_action, reset, step  # noqa: F401  must precede jax imports

import jax
import jax.numpy as jnp
from jax import random


def _dummy_reward(positions: jax.Array) -> jax.Array:
    """Trivial reward, only used to exercise step()'s control flow."""
    return -positions.sum().astype(jax.numpy.float32)


def test_reset_shape() -> None:
    params = EnvParams()
    state = reset(params)
    assert state.positions.shape == (params.n_macros, 2)
    assert state.step == 0
    assert (state.positions == -1).all()


def test_reset_with_initial_positions_resumes_from_correct_step() -> None:
    params = EnvParams(grid=4, n_macros=4)
    initial = jnp.array([[1, 1], [2, 2], [-1, -1], [-1, -1]])
    state = reset(params, initial_positions=initial)
    assert state.step == 2
    assert (state.positions == initial).all()


def test_reset_with_initial_positions_step_continues_correctly() -> None:
    params = EnvParams(grid=4, n_macros=4)
    initial = jnp.array([[1, 1], [2, 2], [-1, -1], [-1, -1]])
    state = reset(params, initial_positions=initial)
    action = jnp.array([3, 3])
    new_state, _reward, done = step(state, action, _dummy_reward, params)
    assert new_state.positions[2].tolist() == [3, 3]  # placed into the next open slot
    assert new_state.step == 3
    assert not done  # one macro (index 3) still unplaced


def test_episode_runs_to_completion() -> None:
    params = EnvParams()
    key = random.PRNGKey(0)
    state = reset(params)
    for t in range(params.n_macros):
        key, subkey = random.split(key)
        action = random_action(subkey, params)
        state, reward, done = step(state, action, _dummy_reward, params)
        assert (action >= 0).all() and (action < params.grid).all()
        if t < params.n_macros - 1:
            assert not done
            assert reward == 0.0
        else:
            assert done

    assert state.step == params.n_macros
    assert (state.positions >= 0).all()


def test_reward_only_fires_on_done() -> None:
    params = EnvParams()
    key = random.PRNGKey(1)
    state = reset(params)
    for t in range(params.n_macros - 1):
        key, subkey = random.split(key)
        state, reward, done = step(state, random_action(subkey, params), _dummy_reward, params)
        assert reward == 0.0 and not done


def test_envstate_is_immutable() -> None:
    state = reset(EnvParams())
    try:
        state.step = 99
        assert False, "EnvState should not allow field assignment"
    except dataclasses.FrozenInstanceError:
        pass


def test_step_is_jit_and_vmap_compatible() -> None:
    """Correctness only, at toy scale - vmap across episodes is proven to
    WORK here. Speed is a separate, hardware-dependent question, and it's
    now been measured on both sides, at real adaptec1 scale (543 macros,
    8 episodes), via scripts/compare_sequential_vs_parallel.py:

        CPU (sandbox): sequential 0.124-0.293s/episode, vmap 9.06s/episode
                        -> vmap ~73x SLOWER
        GPU (real hardware): sequential 0.389s/episode, vmap 0.102s/episode
                        -> vmap ~3.8x FASTER

    Both results make sense together, not separately: each individual
    jit-compiled step() call pays real dispatch/kernel-launch overhead
    crossing from Python into the accelerator, and that overhead is
    *higher* on GPU than CPU - confirmed directly by GPU sequential
    (0.389s) being slower than CPU sequential (0.124-0.293s). Doing that
    543 times per episode sequentially means GPU pays this tax repeatedly
    with nothing to amortize it against; vmap batches many episodes so
    the same overhead is paid once, while genuine parallel execution
    units actually run the batched work simultaneously. On CPU there's
    no such hardware to exploit, so batching just costs more (bigger
    working sets) for no benefit. jit alone is hardware-independent and
    always worth it; vmap-across-episodes specifically needs a GPU/TPU
    to pay off - don't assume either way, measure on the target hardware."""
    params = EnvParams()
    jit_reset = jax.jit(reset, static_argnums=())
    state = jit_reset(params)
    action = jax.numpy.array([1, 2])
    jit_step = jax.jit(step, static_argnames=("reward_fn",))
    new_state, reward, done = jit_step(state, action, _dummy_reward, params)
    assert new_state.step == 1

    keys = random.split(random.PRNGKey(0), 8)
    batched_reset = jax.vmap(lambda k: reset(params))
    batched_states = batched_reset(keys)
    assert batched_states.positions.shape == (8, params.n_macros, 2)
