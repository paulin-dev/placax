"""Compares sequential vs vmap-parallel episode execution speed on real
benchmark data. On CPU-only hardware, sequential+jit was found to be
~73x FASTER than vmap-across-episodes - vmap's speed advantage needs
real parallel hardware. Run this on a GPU to check whether vmap pays
off on your own hardware.

Usage: python scripts/compare_sequential_vs_parallel.py [benchmark_dir] [n_episodes]
"""
import pathlib
import sys
import time

from placax.core import random_action, reset, step  # noqa: F401  must precede jax imports
from placax.netlist import load_netlist  # noqa: F401
from placax.netlist.padding import build_padded_arrays  # noqa: F401
from placax.extras.rewards import make_hpwl_reward  # noqa: F401
from placax.types import EnvParams  # noqa: F401

import jax
import jax.numpy as jnp
from jax import random


def run_sequential(n_episodes: int, params: EnvParams, reward_fn) -> float:
    """N episodes, one after another, each step jit-compiled."""
    jitted_step = jax.jit(step, static_argnames=("reward_fn",))
    jitted_random_action = jax.jit(random_action)

    # warm-up (compile once, outside the timed region)
    state = reset(params)
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    action = jitted_random_action(subkey, params)
    state, _reward, _done = jitted_step(state, action, reward_fn, params)
    jax.block_until_ready(state)

    t0 = time.perf_counter()
    for episode in range(n_episodes):
        state = reset(params)
        key = random.PRNGKey(episode)
        for _ in range(params.n_macros):
            key, subkey = random.split(key)
            action = jitted_random_action(subkey, params)
            state, _reward, _done = jitted_step(state, action, reward_fn, params)
    jax.block_until_ready(state)
    return time.perf_counter() - t0


def run_parallel(n_episodes: int, params: EnvParams, reward_fn) -> float:
    """N episodes at once, batched via vmap - one single jitted function
    per step, not jitted pieces glued together by unjitted code (that
    mistake alone cost an extra ~3.3s in earlier testing)."""

    def one_batched_step(keys, batched_state):
        keys, subkeys = jax.vmap(lambda k: tuple(random.split(k)))(keys)
        actions = jax.vmap(random_action, in_axes=(0, None))(subkeys, params)
        new_state, reward, done = jax.vmap(step, in_axes=(0, 0, None, None))(
            batched_state, actions, reward_fn, params
        )
        return keys, new_state, reward, done

    jitted_batched_step = jax.jit(one_batched_step)

    def make_batch():
        return jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (n_episodes,) + x.shape), reset(params)
        )

    # warm-up
    batched_state = make_batch()
    keys = random.split(random.PRNGKey(0), n_episodes)
    keys, batched_state, _reward, _done = jitted_batched_step(keys, batched_state)
    jax.block_until_ready(batched_state)

    batched_state = make_batch()
    keys = random.split(random.PRNGKey(0), n_episodes)
    t0 = time.perf_counter()
    for _ in range(params.n_macros):
        keys, batched_state, _reward, _done = jitted_batched_step(keys, batched_state)
    jax.block_until_ready(batched_state)
    return time.perf_counter() - t0


if __name__ == "__main__":
    from placax._device import recommended_parallelism_mode

    benchmark_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/adaptec1")
    n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    print(f"JAX backend: {jax.default_backend()}")  # cpu, gpu, or tpu
    print(f"placax.recommended_parallelism_mode(): {recommended_parallelism_mode()!r}")
    print(f"loading {benchmark_dir}...")
    macro_sizes, nets = load_netlist(benchmark_dir)
    _name_to_idx, _sizes, padded_pin_idx, padded_pin_offset, valid_mask = build_padded_arrays(
        macro_sizes, nets
    )
    reward_fn = make_hpwl_reward(padded_pin_idx, padded_pin_offset, valid_mask)
    params = EnvParams(grid=64, n_macros=len(macro_sizes))
    print(f"{params.n_macros} macros, {padded_pin_idx.shape[0]} nets, grid={params.grid}")
    print()

    seq_time = run_sequential(n_episodes, params, reward_fn)
    print(f"sequential ({n_episodes} episodes, jit only): {seq_time:.3f}s")

    par_time = run_parallel(n_episodes, params, reward_fn)
    print(f"parallel   ({n_episodes} episodes, vmap+jit):  {par_time:.3f}s")

    print()
    if par_time < seq_time:
        print(f"vmap is {seq_time / par_time:.1f}x FASTER - parallel hardware paying off")
    else:
        print(f"vmap is {par_time / seq_time:.1f}x SLOWER - matches the CPU-only finding")
