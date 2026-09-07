"""Run provenance and the project's determinism contract.

Two jobs, both in service of the one property this project exists to provide - that two runs
can be compared because everything about them except the thing under test was identical.

1. `fingerprint()` records what a result was produced BY: code version, library versions,
   backend, device, and the dtype/determinism switches that change numbers. A metrics file
   without this is not attributable to a configuration, so every run writes one.

2. `deterministic_backend()` answers whether bit-exact reproduction is actually available
   here. It usually is not, and pretending otherwise is worse than saying so:

   **JAX on GPU is not run-to-run deterministic in this project.** Two identical runs (same
   seed, same process, same machine) diverge - measured at ~1e-16 by the second PPO iteration
   and ~1e-10 by the fifth, growing from there. It is localized to the backward pass: forward
   passes and rollouts reproduce exactly, but repeated calls to the same jitted
   `jax.grad(ppo_loss)` on identical inputs return convolution gradients differing by up to
   7.5e-9. Neither `--xla_gpu_deterministic_ops=true` nor
   `--xla_gpu_exclude_nondeterministic_ops=true` removes it. The CPU backend IS bit-exact,
   and CPU and GPU also disagree with each other, so results are not comparable across
   backends either.

   The consequences, which callers are expected to honor:
     - A single GPU run is not a reproducible result. Report across seeds, not from one run.
     - Any claim of bit-exactness (checkpoint resume, refactor equivalence) must either run
       on a deterministic backend or be stated as "within the backend's own noise floor" -
       see `measure_noise_floor` in the test suite for how that floor is established.
     - Set PLACAX_DETERMINISTIC=1 to force the CPU backend when bit-exactness matters more
       than speed. That is the only switch here that actually delivers it.
"""
import importlib.metadata
import os
import pathlib
import platform
import subprocess

from placax import _device  # noqa: F401  must run before any `import jax` below

_RECORDED_PACKAGES = ("jax", "jaxlib", "flax", "optax", "numpy", "placax")


def deterministic_backend() -> bool:
    """True if this process's JAX backend reproduces bit-exactly run to run.

    Only the CPU backend qualifies today - see this module's docstring for the measurement.
    """
    import jax

    return jax.default_backend() == "cpu"


def _git_revision(repo_root: pathlib.Path) -> tuple[str | None, bool | None]:
    """(short SHA, dirty) for repo_root, or (None, None) if it isn't a git checkout / git is absent."""
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True, text=True, check=True, timeout=10,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    revision = git("rev-parse", "--short", "HEAD")
    if revision is None:
        return None, None
    # An empty `status --porcelain` means every tracked file matches HEAD, i.e. the recorded
    # SHA really does describe the code that ran.
    status = git("status", "--porcelain")
    return revision, (bool(status) if status is not None else None)


def _package_versions() -> dict[str, str | None]:
    """Installed version of each package whose behavior can move the numbers."""
    versions: dict[str, str | None] = {}
    for name in _RECORDED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _device_description() -> dict:
    """Backend, device kind, and device count as JAX itself reports them."""
    import jax

    devices = jax.devices()
    return {
        "backend": jax.default_backend(),
        "device_kind": devices[0].device_kind if devices else None,
        "device_count": len(devices),
        "platform": devices[0].platform if devices else None,
    }


def fingerprint(repo_root: pathlib.Path | None = None) -> dict:
    """Everything about this machine and build that can change a result, as a JSON-able dict.

    Written into every run's manifest. Compare two manifests' fingerprints before comparing
    their metrics: a differing backend or jax version is enough to explain a differing number
    on its own, and knowing that up front saves chasing it as if it were a real effect.
    """
    # Default to the checkout this file lives in, which is the code that actually ran.
    root = repo_root or pathlib.Path(__file__).resolve().parent.parent
    revision, dirty = _git_revision(root)
    return {
        "git_revision": revision,
        "git_dirty": dirty,
        "packages": _package_versions(),
        "device": _device_description(),
        "deterministic_backend": deterministic_backend(),
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        # Both of these change numeric results directly, and both are set by _device.py from
        # the environment, so they belong in the record rather than being assumed.
        "jax_enable_x64": os.environ.get("JAX_ENABLE_X64"),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "placax_deterministic": os.environ.get("PLACAX_DETERMINISTIC"),
    }


def describe_determinism() -> str:
    """One line for a run's console header, so the guarantee in force is never implicit."""
    if deterministic_backend():
        return "backend is bit-exact run to run (CPU)"
    return (
        "backend is NOT bit-exact run to run (GPU conv gradients vary; see "
        "placax.reproducibility) - compare across seeds, not single runs; "
        "set PLACAX_DETERMINISTIC=1 to force the deterministic CPU backend"
    )
