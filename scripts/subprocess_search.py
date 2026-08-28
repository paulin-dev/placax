"""Generic subprocess parameter sweep - deliberately imports nothing from placax/placax_agents/jax,
only the standard library.

Why that matters: every module under placax/placax_agents transitively imports placax/__init__.py,
which - as of this writing - eagerly touches JAX's backend at *import* time (a GPU client gets
initialized and, under JAX's default allocator, preallocates a large fraction of the whole device
as one arena, immediately, before any training code runs). A driver process built on that import
chain is stuck holding that reservation for its entire lifetime, so it can never accurately probe
disposable subprocesses for how much GPU memory they need - they'd be competing with their own
already-reserving parent for the same physical device. The fix is making sure the process driving
the search never imports anything that touches the GPU in the first place - this module doesn't,
so run it as its own separate invocation, before starting whatever it's probing.

No cooperation is required from the target script either - it's just run with its own, completely
ordinary CLI, once per candidate value of one flag. Success is exit code 0; a nonzero exit whose
output looks like an out-of-memory error is treated as "that value doesn't fit"; anything else is
a real bug in the target and is raised immediately, not silently swallowed.

Usage: exactly one argument must be --name=[v1,v2,...] (the flag to sweep); everything else is
passed through unchanged to the target on every attempt - use a short --n_iterations (or
whatever keeps one attempt fast) and, if the target supports it, a flag to skip checkpointing, so
repeated attempts stay fast and don't leak state into each other.

    python -m scripts.subprocess_search scripts.run_maskplace '--n_episodes=[1,2,4,8,10]' \\
        --benchmark_dir=benchmarks/adaptec1 --macro_budget=128 --eval_every=1 --n_iterations=4 --no_checkpoint

(quote the swept flag - most shells would otherwise glob-expand the unquoted brackets. The
--eval_every=1/--n_iterations=4 there are run_maskplace-specific tuning, not a rule of this tool -
see that script's own docstring for why: a probe should cross at least one --eval_every boundary,
since it's a separately-compiled path with its own memory footprint, but doesn't need to match the
real run's --eval_every to do that.)

Tries values in the given order and stops at the first failure (so list them smallest-first),
then prints "RESULT=<largest value that worked>" (or "RESULT=none").
"""
import subprocess
import sys
import time

_DEFAULT_TIMEOUT_S = 900.0  # generous: cold JIT compilation, not steady-state speed, usually dominates


def _looks_like_oom(text: str) -> bool:
    lowered = text.lower()
    return (
        "resource_exhausted" in lowered
        or "out of memory" in lowered
        or "outofmemoryerror" in lowered
        or "memoryerror" in lowered
        or "cannot allocate memory" in lowered
    )


def try_value(module: str, name: str, value: str, other_args: list[str], timeout_s: float) -> bool:
    """Runs `python -m module --name=value *other_args` once; True if it exits 0, False if it
    looks like an OOM. Any other failure (a real bug in the target) raises RuntimeError."""
    argv = [sys.executable, "-m", module, f"--{name}={value}", *other_args]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False  # a hang is very unlikely to ever finish - treat it the same as infeasible

    if result.returncode == 0:
        return True
    output = result.stdout + result.stderr
    if result.returncode < 0 or _looks_like_oom(output):  # returncode<0: killed by a signal, e.g. an
        return False                                       # allocator hitting a fatal internal check
    raise RuntimeError(f"{module} --{name}={value} crashed (exit {result.returncode}), not an OOM:\n{output[-4000:]}")


def sweep(
    module: str, name: str, values: list[str], other_args: list[str], timeout_s: float = _DEFAULT_TIMEOUT_S
) -> str | None:
    """Tries each value in order, stopping at the first failure. Returns the largest value that
    worked, or None if even the first one didn't."""
    last_good = None
    for value in values:
        print(f"Trying {name}={value}...", file=sys.stderr)
        start = time.monotonic()
        ok = try_value(module, name, value, other_args, timeout_s)
        print(f"  {name}={value}: {'ok' if ok else 'failed'} ({time.monotonic() - start:.1f}s)", file=sys.stderr)
        if not ok:
            break
        last_good = value
    return last_good


def _parse_argv(argv: list[str]) -> tuple[str, str, list[str], list[str], float]:
    """Returns (module, swept_name, values, other_args, timeout_s) from sys.argv[1:]. Exactly one
    argument must be --name=[v1,v2,...]; an optional --timeout=SECONDS may appear anywhere else."""
    if not argv:
        raise SystemExit("usage: python -m scripts.subprocess_search <module> --name=[v1,v2,...] [other args...]")
    module, args = argv[0], argv[1:]

    timeout_s = _DEFAULT_TIMEOUT_S
    rest = []
    for arg in args:
        if arg.startswith("--timeout="):
            timeout_s = float(arg.removeprefix("--timeout="))
        else:
            rest.append(arg)

    sweep_index = next((i for i, a in enumerate(rest) if a.startswith("--") and "=[" in a and a.endswith("]")), None)
    if sweep_index is None:
        raise SystemExit("need exactly one --name=[v1,v2,...] argument to sweep.")
    flag, _, raw_values = rest[sweep_index].partition("=")
    name = flag.removeprefix("--")
    values = [v.strip() for v in raw_values.strip("[]").split(",") if v.strip()]
    other_args = rest[:sweep_index] + rest[sweep_index + 1 :]
    return module, name, values, other_args, timeout_s


def main() -> None:
    module, name, values, other_args, timeout_s = _parse_argv(sys.argv[1:])
    result = sweep(module, name, values, other_args, timeout_s=timeout_s)
    # The only thing this tool ever prints to stdout - every progress line above went to stderr,
    # so a caller can capture this directly: N=$(python -m scripts.subprocess_search ...)
    print(f"RESULT={result if result is not None else 'none'}")


if __name__ == "__main__":
    main()
