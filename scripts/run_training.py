"""Runs (or resumes) an extended training session on a real benchmark,
checkpointing and evaluating real HPWL along the way. This is the
script tying together everything built as library pieces - resumable_train,
checkpointing, real evaluation, and autotuning - into one thing you can
actually run.

Usage:
    python scripts/run_training.py [benchmark_dir] [n_iterations] [n_envs]

Example:
    python scripts/run_training.py benchmarks/adaptec1 200          # auto-detects everything
    python scripts/run_training.py benchmarks/adaptec1 200 32       # force n_envs=32 explicitly

n_envs, if not given, is genuinely auto-detected: recommended_parallelism_mode()
first decides whether parallel is even worth trying on this hardware at
all, and only if so does find_max_batch_size() search for the largest
batch size that fits.

That search calls train_parallel() directly - the exact same public
function real training uses, with n_iterations=1 - on the exact same
real benchmark data, in this process. No subprocess, no separate toy
example, no hand-rolled approximation of what a real step does: each
candidate genuinely IS one real training step at that batch size, using
the real code path, so what the search measures is exactly what real
training will actually do.

This is a deliberate simplification over an earlier version that ran
each candidate in an isolated subprocess with a hard timeout, to
protect against XLA's own out-of-memory retry logic hanging
indefinitely on a bad candidate (confirmed directly: 4+ minutes stuck,
no exception, no progress). That protection is real, but so is the
cost: subprocess spawning is slow, and testing via a hand-rolled
approximation of a step - rather than the real train_parallel() call -
risks measuring something subtly different from what real training
actually does. This version trades the automatic-hang-protection for
directness and simplicity: if a candidate genuinely hangs, this script
will hang too, and you'll need to interrupt it (Ctrl+C) and re-run with
a smaller max_candidate or an explicit n_envs. That tradeoff is
deliberate, not an oversight.

Safe to re-run: it automatically resumes from checkpoint.bin in the
benchmark's own directory if one already exists, rather than starting
over - point it at a GPU machine with the same checkpoint file copied
over, and it picks up exactly where it left off. Every 50 iterations it
also saves a permanent, never-overwritten snapshot (checkpoint_iter_N.bin
in a snapshots/ subfolder), so you can roll back to an earlier point if
a later run goes wrong - checkpoint.bin itself is always overwritten,
so without a snapshot you'd have no way back once training moves past it.
"""
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

import gc
import jax
from jax import random


def _jax_cleanup() -> None:
    """Releases what's safe to release between candidates, without
    touching anything still in use.

    jax.clear_caches() only clears compiled-executable caches, never
    data - always safe. gc.collect() only frees objects with zero
    Python references - also always safe, since it can't touch
    anything still referenced.

    Deliberately does NOT call jax.live_arrays() + .delete(): an
    earlier version did, copying a pattern from an isolated example
    without checking it fit this use case - it deletes EVERY live JAX
    array indiscriminately, with no way to tell "transient scratch from
    this one attempt" from "a persistent input the next attempt still
    needs". Confirmed as a real, severe bug: it deleted
    valid_mask/padded_pin_idx/padded_pin_offset (closed over by
    reward_fn, reused across every candidate) mid-search, crashing
    training on the very next attempt with "Array has been deleted"."""
    jax.clear_caches()
    gc.collect()


def _auto_detect_n_envs(
    key, variables, policy, params, reward_fn, sizes_array, cell_size, max_candidate: int = 64
) -> tuple[str, int]:
    """Returns (mode, n_envs). Only searches at all if
    recommended_parallelism_mode() says parallel is worth trying on
    this hardware - no point searching for a batch size we're not
    going to use.

    Each candidate is one real call to train_parallel(..., n_iterations=1)
    on the real benchmark - not a synthetic approximation - so what
    this measures is exactly what real training will do at that n_envs.
    max_candidate defaults to 64, not 256: a real training step is far
    more expensive to test than a bare forward pass, and a smaller
    ceiling means less time spent testing genuinely oversized
    candidates that were never going to work anyway."""
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
