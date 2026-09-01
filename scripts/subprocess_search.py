"""Generic subprocess parameter sweep, deliberately dependency-free of placax/jax so it never reserves GPU memory itself."""
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
    """Runs `python -m module --name=value *other_args` once; True on success, False on OOM, else raises."""
    argv = [sys.executable, "-m", module, f"--{name}={value}", *other_args]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False  # a hang is very unlikely to ever finish - treat it the same as infeasible

    if result.returncode == 0:
        return True
    output = result.stdout + result.stderr
    if result.returncode < 0 or _looks_like_oom(output):  # returncode<0: killed by a signal (e.g. fatal alloc check)
        return False
    raise RuntimeError(f"{module} --{name}={value} crashed (exit {result.returncode}), not an OOM:\n{output[-4000:]}")


def sweep(
    module: str, name: str, values: list[str], other_args: list[str], timeout_s: float = _DEFAULT_TIMEOUT_S
) -> str | None:
    """Tries each value in order, stopping at the first failure; returns the largest value that worked."""
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
    """Returns (module, swept_name, values, other_args, timeout_s) from sys.argv[1:]."""
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
    # The only thing this tool prints to stdout, so a caller can capture it directly.
    print(f"RESULT={result if result is not None else 'none'}")


if __name__ == "__main__":
    main()
