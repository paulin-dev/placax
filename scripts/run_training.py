"""Runs (or resumes) extended training on a real benchmark, checkpointing
and evaluating real HPWL along the way - the script tying the library
pieces together into one thing you can actually run.

Usage:
    python scripts/run_training.py [benchmark_dir] [n_iterations] [n_envs]

n_envs auto-detection tries each candidate for real, but in a disposable
subprocess (see _try_n_envs_subprocess), not this process. Two reasons:

1. JAX preallocates ~75-90% of GPU memory as one arena on first use and
   never gives it back, so this process's own memory_stats() can't tell
   candidates apart after the first one runs - every number afterward
   reflects that one big reservation, not what any given n_envs actually
   needs. A fresh process hasn't made that reservation yet.
2. A candidate that hangs or crashes the GPU driver (observed directly
   while building this) can't reliably be interrupted with SIGTERM
   in-process. A subprocess can be given a hard wall-clock timeout and
   killed outright without losing the training run driving the search.

Safe to re-run: resumes from checkpoint.bin automatically; every 50
iterations a permanent snapshot (never overwritten) is saved under
snapshots/ for rollback."""
import os
import pathlib
import subprocess
import sys

from placax._device import recommended_parallelism_mode  # noqa: F401  must precede jax imports
from placax.core import reset  # noqa: F401
from placax.netlist import load_netlist  # noqa: F401
from placax.netlist.padding import build_padded_arrays  # noqa: F401
from placax.types import EnvParams  # noqa: F401
from placax_agents.ops.autotune import find_max_batch_size, is_oom  # noqa: F401
from placax_agents.ops.resumable_train import resumable_train  # noqa: F401
from placax_agents.policy.architectures.cnn import CNNActorCritic  # noqa: F401
from placax_agents.policy.observation import observation  # noqa: F401
from placax_agents.policy.scale import compute_grid_scale  # noqa: F401
from placax_agents.training.loops.parallel_train import train_parallel  # noqa: F401
from placax_agents.training.reward import make_scaled_hpwl_reward  # noqa: F401

from jax import random

_PROBE_FLAG = "--probe-n-envs"
_PROBE_OOM_MARKER = "PLACAX_PROBE_OOM"
_PROBE_TIMEOUT_S = 90.0


def _build_training_state(benchmark_dir: pathlib.Path):
    """Loads the netlist and builds everything needed to attempt a
    training step: policy, initial variables, reward_fn, sizes_array,
    cell_size, params. Shared between real training and the n_envs
    probe subprocess so both run the exact same computation."""
    macro_sizes, nets = load_netlist(benchmark_dir)
    _, sizes_array, padded_pin_idx, padded_pin_offset, valid_mask = build_padded_arrays(
        macro_sizes, nets
    )
    params = EnvParams(grid=64, n_macros=len(macro_sizes))
    cell_size = compute_grid_scale(sizes_array, params.grid_x, params.effective_grid_y)
    reward_fn = make_scaled_hpwl_reward(
        padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size
    )

    policy = CNNActorCritic()
    key = random.PRNGKey(0)
    key, init_key = random.split(key)
    obs0 = observation(reset(params), params, sizes_array)
    variables_init = policy.init(init_key, obs0)

    return {
        "macro_sizes": macro_sizes, "nets": nets, "params": params, "cell_size": cell_size,
        "reward_fn": reward_fn, "sizes_array": sizes_array, "padded_pin_idx": padded_pin_idx,
        "padded_pin_offset": padded_pin_offset, "valid_mask": valid_mask,
        "policy": policy, "key": key, "variables_init": variables_init,
    }


def _probe_entrypoint(benchmark_dir: pathlib.Path, n: int) -> None:
    """Run as a subprocess of _try_n_envs_subprocess: attempts one real
    n-env training step and reports the outcome via exit code, so the
    parent never has to interpret this process's internals - just
    whether it succeeded, hit a real OOM, or crashed for some other
    reason."""
    state = _build_training_state(benchmark_dir)
    try:
        train_parallel(
            state["key"], state["variables_init"], state["policy"].apply, state["params"],
            state["reward_fn"], state["sizes_array"], state["cell_size"], n_envs=n, n_iterations=1,
        )
    except Exception as e:
        if is_oom(e):
            print(_PROBE_OOM_MARKER)
            sys.exit(1)
        raise


def _try_n_envs_subprocess(benchmark_dir: pathlib.Path, n: int) -> None:
    """try_fn for find_max_batch_size: attempts n_envs=n in a fresh,
    disposable process instead of this one (see module docstring for
    why). Raises MemoryError on OOM or timeout - either way, n_envs=n
    doesn't work here - so find_max_batch_size backs off exactly as it
    would for an in-process OOM. Any other failure is a real bug and
    propagates with the subprocess's traceback attached."""
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    try:
        result = subprocess.run(
            [sys.executable, __file__, _PROBE_FLAG, str(benchmark_dir), str(n)],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S, env=env,
        )
    except subprocess.TimeoutExpired as e:
        raise MemoryError(f"n_envs={n} probe exceeded {_PROBE_TIMEOUT_S:.0f}s, treating as infeasible") from e

    if result.returncode == 0:
        return
    if _PROBE_OOM_MARKER in result.stdout:
        raise MemoryError(f"n_envs={n} does not fit")
    raise RuntimeError(f"probe for n_envs={n} crashed (exit {result.returncode}):\n{result.stderr[-4000:]}")


def _auto_detect_n_envs(benchmark_dir: pathlib.Path, max_candidate: int = 64) -> tuple[str, int]:
    """Returns (mode, n_envs). Only searches if parallel is worth trying
    on this hardware at all."""
    mode = recommended_parallelism_mode()
    if mode == "sequential":
        return "sequential", 1

    n_envs = find_max_batch_size(
        lambda n: _try_n_envs_subprocess(benchmark_dir, n), max_candidate=max_candidate
    )
    return "parallel", max(n_envs, 1)


def _parse_args(argv: list[str]) -> tuple[pathlib.Path, int, int | None]:
    """(benchmark_dir, n_iterations, n_envs_override) from sys.argv."""
    benchmark_dir = pathlib.Path(argv[1] if len(argv) > 1 else "benchmarks/adaptec1")
    n_iterations = int(argv[2]) if len(argv) > 2 else 100
    n_envs_override = int(argv[3]) if len(argv) > 3 else None
    return benchmark_dir, n_iterations, n_envs_override


def _resolve_n_envs(benchmark_dir: pathlib.Path, n_envs_override: int | None) -> tuple[str, int]:
    """(mode, n_envs): the override if given, else auto-detected. Prints
    its own progress since auto-detection can take a while (see
    _auto_detect_n_envs)."""
    if n_envs_override is not None:
        mode = "sequential" if n_envs_override <= 1 else "parallel"
        print(f"n_envs={n_envs_override} given explicitly, mode={mode}")
        return mode, n_envs_override

    print("auto-detecting mode and n_envs (probing candidates in disposable subprocesses) ...")
    mode, n_envs = _auto_detect_n_envs(benchmark_dir)
    print(f"  -> mode={mode}, n_envs={n_envs}")
    return mode, n_envs


def _print_results(
    log: list[dict], checkpoint_path: pathlib.Path, snapshot_dir: pathlib.Path, log_path: pathlib.Path
) -> None:
    print()
    print("iteration    loss           real_hpwl")
    for entry in log:
        hpwl_str = f"{entry['real_hpwl']:.1f}" if entry["real_hpwl"] is not None else "-"
        print(f"{entry['iteration']:>9}    {entry['loss']:>10.4f}    {hpwl_str}")

    print()
    print(f"checkpoint saved to {checkpoint_path} - re-run this script to continue training.")
    print(f"snapshots (never overwritten) saved to {snapshot_dir}/")
    print(f"full history saved to {log_path}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == _PROBE_FLAG:
        _probe_entrypoint(pathlib.Path(sys.argv[2]), int(sys.argv[3]))
        return

    benchmark_dir, n_iterations, n_envs_override = _parse_args(sys.argv)
    if not benchmark_dir.exists():
        print(f"'{benchmark_dir}' not found - run scripts/download_benchmarks.py first.")
        sys.exit(1)

    print(f"loading {benchmark_dir} ...")
    state = _build_training_state(benchmark_dir)
    print(f"  {len(state['macro_sizes'])} macros, {len(state['nets'])} nets, cell_size={state['cell_size']:.2f}")

    checkpoint_path = benchmark_dir / "checkpoint.bin"
    snapshot_dir = benchmark_dir / "snapshots"
    log_path = benchmark_dir / "training_log.jsonl"
    mode, n_envs = _resolve_n_envs(benchmark_dir, n_envs_override)

    resuming = checkpoint_path.exists()
    print(f"{'resuming from' if resuming else 'starting fresh, will save to'} {checkpoint_path}")
    print(f"running {n_iterations} more iterations ...")

    _final_variables, log = resumable_train(
        checkpoint_path, state["variables_init"], state["key"], state["policy"].apply, state["params"],
        state["reward_fn"], state["sizes_array"], state["cell_size"], n_iterations,
        state["padded_pin_idx"], state["padded_pin_offset"], state["valid_mask"],
        checkpoint_every=10, eval_every=10, log_path=log_path,
        n_envs=n_envs, mode=mode, snapshot_dir=snapshot_dir, snapshot_every=50,
    )
    _print_results(log, checkpoint_path, snapshot_dir, log_path)


if __name__ == "__main__":
    main()
