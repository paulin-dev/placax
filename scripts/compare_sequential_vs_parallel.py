"""Compares sequential vs vmap-parallel episode execution speed on real benchmark data."""
import pathlib
import sys
import time

from placax.core import random_action, reset, step  # must precede jax imports
from placax.extras.rewards import make_hpwl_reward
from placax.log import Log
from placax.netlist import load_netlist
from placax.netlist.padding import build_padded_arrays
from placax.types import EnvParams

import jax
import jax.numpy as jnp
from jax import random


def run_sequential(n_episodes: int, params: EnvParams, reward_fn) -> float:
    """Runs n_episodes one after another, each step jit-compiled, and returns the wall-clock time."""
    # 1. Compile the two hot functions once; JAX caches the compiled version.
    jitted_step = jax.jit(step, static_argnames=("reward_fn",))
    jitted_random_action = jax.jit(random_action)

    # 2. Warm-up: run one step here so jit compilation doesn't pollute the timed region below.
    state = reset(params)
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    action = jitted_random_action(subkey, params)
    state, _reward, _done = jitted_step(state, action, reward_fn, params)
    jax.block_until_ready(state)

    # 3. Now time the real work: n_episodes, each stepping through every macro one at a time.
    t0 = time.perf_counter()
    for episode in range(n_episodes):
        state = reset(params)
        key = random.PRNGKey(episode)
        for _ in range(params.n_macros):
            key, subkey = random.split(key)
            action = jitted_random_action(subkey, params)
            state, _reward, _done = jitted_step(state, action, reward_fn, params)
    # block_until_ready forces JAX's async dispatch to finish before we stop the clock.
    jax.block_until_ready(state)
    return time.perf_counter() - t0


def run_parallel(n_episodes: int, params: EnvParams, reward_fn) -> float:
    """Runs n_episodes at once via vmap, as a single jitted step function, and returns the wall-clock time."""
    def one_batched_step(keys, batched_state):
        # Same step() as sequential, just vmapped over the leading batch dimension.
        keys, subkeys = jax.vmap(lambda k: tuple(random.split(k)))(keys)
        actions = jax.vmap(random_action, in_axes=(0, None))(subkeys, params)
        new_state, reward, done = jax.vmap(step, in_axes=(0, 0, None, None))(
            batched_state, actions, reward_fn, params
        )
        return keys, new_state, reward, done

    # 1. Compile the batched step once; every episode in the batch runs it in lockstep.
    jitted_batched_step = jax.jit(one_batched_step)

    def make_batch():
        # Turn one fresh env state into n_episodes identical copies stacked along a new "batch" axis.
        return jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (n_episodes,) + x.shape), reset(params)
        )

    # 2. Warm-up: pay the jit compile cost here, outside the timed region.
    batched_state = make_batch()
    keys = random.split(random.PRNGKey(0), n_episodes)
    keys, batched_state, _reward, _done = jitted_batched_step(keys, batched_state)
    jax.block_until_ready(batched_state)

    # 3. Now time the real work: n_macros steps, each advancing the whole batch at once.
    batched_state = make_batch()
    keys = random.split(random.PRNGKey(0), n_episodes)
    t0 = time.perf_counter()
    for _ in range(params.n_macros):
        keys, batched_state, _reward, _done = jitted_batched_step(keys, batched_state)
    jax.block_until_ready(batched_state)
    return time.perf_counter() - t0


if __name__ == "__main__":
    from placax._device import recommended_parallelism_mode

    Log.configure()

    benchmark_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/adaptec1")
    n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    # 1. Report the hardware/backend so results are reproducible/comparable.
    Log.info(f"JAX backend: {jax.default_backend()}")
    Log.info(f"placax.recommended_parallelism_mode(): {recommended_parallelism_mode()!r}")

    # 2. Load the real benchmark netlist, pad it into fixed-size arrays, and build the reward function.
    Log.info(f"loading {benchmark_dir}...")
    macro_sizes, nets = load_netlist(benchmark_dir)
    _name_to_idx, _sizes, padded_pin_idx, padded_pin_offset, valid_mask = build_padded_arrays(
        macro_sizes, nets
    )
    reward_fn = make_hpwl_reward(padded_pin_idx, padded_pin_offset, valid_mask)
    params = EnvParams(grid=64, n_macros=len(macro_sizes))
    Log.info(f"{params.n_macros} macros, {padded_pin_idx.shape[0]} nets, grid={params.grid}")

    # 3. Run both strategies on the exact same benchmark and compare timings.
    seq_time = run_sequential(n_episodes, params, reward_fn)
    par_time = run_parallel(n_episodes, params, reward_fn)

    # Results, not log messages - printed as a clean report.
    print()
    print(f"sequential ({n_episodes} episodes, jit only): {seq_time:.3f}s")
    print(f"parallel   ({n_episodes} episodes, vmap+jit):  {par_time:.3f}s")
    print()
    if par_time < seq_time:
        print(f"vmap is {seq_time / par_time:.1f}x FASTER - parallel hardware paying off")
    else:
        print(f"vmap is {par_time / seq_time:.1f}x SLOWER - matches the CPU-only finding")
