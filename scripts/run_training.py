"""Runs (or resumes) extended training on a real benchmark, checkpointing
and evaluating real HPWL along the way - the script tying the library
pieces together into one thing you can actually run.

Usage:
    python scripts/run_training.py [benchmark_dir] [n_iterations] [n_envs]

n_envs auto-detection runs real 1-step train_parallel() trials on the
real benchmark in this process (no subprocess, no approximation), so
what's measured is exactly what training will do. Tradeoff: a candidate
that hangs hangs this script too - interrupt with Ctrl+C and re-run with
a smaller max_candidate or explicit n_envs.

Safe to re-run: resumes from checkpoint.bin automatically; every 50
iterations a permanent snapshot (never overwritten) is saved under
snapshots/ for rollback."""
import gc
import pathlib
import sys

from placax._device import recommended_parallelism_mode  # noqa: F401  must precede jax imports
from placax.core import reset  # noqa: F401
from placax.netlist import load_netlist  # noqa: F401
from placax.netlist.padding import build_padded_arrays  # noqa: F401
from placax.types import EnvParams  # noqa: F401
from placax_agents.ops.autotune import find_max_batch_size  # noqa: F401
from placax_agents.ops.resumable_train import resumable_train  # noqa: F401
from placax_agents.policy.architectures.cnn import CNNActorCritic  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.policy.scale import compute_grid_scale  # noqa: F401
from placax_agents.training.loops.parallel_train import train_parallel  # noqa: F401
from placax_agents.training.reward import make_scaled_hpwl_reward  # noqa: F401

import jax
from jax import random


def _jax_cleanup() -> None:
    """Releases what's safe between autotune candidates: compiled-
    executable caches and unreferenced objects. Deliberately does NOT
    delete live arrays (jax.live_arrays + delete) - that would also kill
    arrays closed over by reward_fn that later candidates still need."""
    jax.clear_caches()
    gc.collect()


def _auto_detect_n_envs(
    key, variables, policy, params, reward_fn, sizes_array, cell_size, max_candidate: int = 64
):
    """Returns (mode, n_envs). Only searches if parallel is worth trying
    on this hardware at all."""
    mode = recommended_parallelism_mode()
    if mode == "sequential":
        return "sequential", 1

    def try_n_envs(n: int) -> None:
        train_parallel(
            key, variables, policy.apply, params, reward_fn, sizes_array, cell_size,
            n_envs=n, n_iterations=1,
        )

    n_envs = find_max_batch_size(try_n_envs, max_candidate=max_candidate, cleanup_fn=_jax_cleanup)
    return "parallel", max(n_envs, 1)


def main() -> None:
    benchmark_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/adaptec1")
    n_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    n_envs_override = int(sys.argv[3]) if len(sys.argv) > 3 else None

    if not benchmark_dir.exists():
        print(f"'{benchmark_dir}' not found - run scripts/download_benchmarks.py first.")
        sys.exit(1)

    print(f"loading {benchmark_dir} ...")
    macro_sizes, nets = load_netlist(benchmark_dir)
    _, sizes_array, padded_pin_idx, padded_pin_offset, valid_mask = build_padded_arrays(
        macro_sizes, nets
    )
    params = EnvParams(grid=64, n_macros=len(macro_sizes))
    cell_size = compute_grid_scale(sizes_array, params.grid_x, params.effective_grid_y)
    reward_fn = make_scaled_hpwl_reward(
        padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size
    )
    print(f"  {len(macro_sizes)} macros, {len(nets)} nets, cell_size={cell_size:.2f}")

    checkpoint_path = benchmark_dir / "checkpoint.bin"
    snapshot_dir = benchmark_dir / "snapshots"
    log_path = benchmark_dir / "training_log.jsonl"

    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = observation(reset(params), params, sizes_array)
    variables_init = policy.init(init_key, obs0)

    if n_envs_override is not None:
        mode = "sequential" if n_envs_override <= 1 else "parallel"
        n_envs = n_envs_override
        print(f"n_envs={n_envs} given explicitly, mode={mode}")
    else:
        print("auto-detecting mode and n_envs (running real 1-step trials, no subprocess) ...")
        key, probe_key = random.split(key)
        mode, n_envs = _auto_detect_n_envs(
            probe_key, variables_init, policy, params, reward_fn, sizes_array, cell_size
        )
        print(f"  -> mode={mode}, n_envs={n_envs}")

    resuming = checkpoint_path.exists()
    print(f"{'resuming from' if resuming else 'starting fresh, will save to'} {checkpoint_path}")
    print(f"running {n_iterations} more iterations ...")

    _final_variables, log = resumable_train(
        checkpoint_path, variables_init, key, policy.apply, params, reward_fn, sizes_array,
        cell_size, n_iterations, padded_pin_idx, padded_pin_offset, valid_mask,
        checkpoint_every=10, eval_every=10, log_path=log_path,
        n_envs=n_envs, mode=mode, snapshot_dir=snapshot_dir, snapshot_every=50,
    )

    print()
    print("iteration    loss           real_hpwl")
    for entry in log:
        hpwl_str = f"{entry['real_hpwl']:.1f}" if entry["real_hpwl"] is not None else "-"
        print(f"{entry['iteration']:>9}    {entry['loss']:>10.4f}    {hpwl_str}")

    print()
    print(f"checkpoint saved to {checkpoint_path} - re-run this script to continue training.")
    print(f"snapshots (never overwritten) saved to {snapshot_dir}/")
    print(f"full history saved to {log_path}")


if __name__ == "__main__":
    main()
