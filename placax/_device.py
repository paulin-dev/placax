"""Import first, before `import jax` anywhere - env vars are read once at JAX's first import."""
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

# No GPU: force JAX onto CPU explicitly, rather than letting it try and fail.
if not _GPU_PHYSICALLY_PRESENT:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

# WSL2's GPU passthrough is memory-constrained - cap JAX's own preallocation.
if _is_wsl():
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.70")

# Silences XLA/cuDNN's C++ logging (routine kernel-search noise, not
# actionable); a real fatal error still aborts the process regardless.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


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
    """Returns "sequential" or "parallel", using override if given, else auto-detecting from the backend."""
    # An explicit override always wins, but must still be one of the two valid modes.
    if override is not None:
        if override not in ("sequential", "parallel"):
            raise ValueError(f"mode must be 'sequential' or 'parallel', got {override!r}")
        return override

    # Otherwise pick based on measured behavior: vmap is much slower than a
    # sequential loop on CPU, but much faster on GPU.
    import jax

    return "sequential" if jax.default_backend() == "cpu" else "parallel"
