"""Import first, before `import jax` anywhere - env vars are read once
at JAX's first import."""
import os
import platform
import subprocess
import warnings


def _gpu_available() -> bool:
    """True if nvidia-smi runs, i.e. a GPU + driver are present."""
    try:
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _is_wsl() -> bool:
    """True under WSL, detected via the kernel release string."""
    return "microsoft" in platform.release().lower()


_GPU_PHYSICALLY_PRESENT: bool = _gpu_available()

if not _GPU_PHYSICALLY_PRESENT:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

if _is_wsl():
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.70")


def warn_if_gpu_unused() -> None:
    """Warns if a GPU is present but JAX fell back to CPU (missing CUDA extra)."""
    import jax

    if _GPU_PHYSICALLY_PRESENT and jax.default_backend() == "cpu":
        warnings.warn(
            "nvidia-smi reports a GPU, but JAX is only using the CPU backend. "
            "Run: pip install 'placax[cuda]' (or 'placax[cuda12]' as a fallback) "
            "to actually use the GPU.",
            stacklevel=2,
        )


def recommended_parallelism_mode(override: str | None = None) -> str:
    """Returns "sequential" or "parallel", from `override` if given,
    else auto-detected from the JAX backend. Measured, not guessed:
    vmap-across-episodes is ~73x slower than sequential+jit on CPU,
    ~3.8x faster on GPU (scripts/compare_sequential_vs_parallel.py)."""
    if override is not None:
        if override not in ("sequential", "parallel"):
            raise ValueError(f"mode must be 'sequential' or 'parallel', got {override!r}")
        return override

    import jax

    return "sequential" if jax.default_backend() == "cpu" else "parallel"
