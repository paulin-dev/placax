"""Helpers for asserting resume-equals-continuous without assuming a deterministic backend.

The property these tests care about is "resuming from a checkpoint changes nothing about the
run". The obvious encoding - assert the interrupted run's numbers equal the continuous run's
exactly - is only valid on a backend that reproduces bit-exactly, and JAX on GPU does not
(placax/reproducibility.py documents the measurement). Those tests therefore failed on every
GPU host, for a reason that had nothing to do with checkpoint resume.

The fix is to compare against a control rather than against zero: run the SAME continuous
configuration twice to measure how far the backend moves on its own, then require the
interrupted run to be at least as close as that. On a deterministic backend the floor is
exactly 0.0 and this stays a strict bit-exactness test; on GPU it stays a real test of resume
(a genuinely broken resume diverges by orders of magnitude more than backend noise) instead of
a test of the hardware.
"""
from placax.reproducibility import deterministic_backend  # noqa: F401  must precede jax imports

import jax


def max_abs_difference(a, b) -> float:
    """Largest absolute elementwise difference between two matching pytrees (or number lists)."""
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    assert len(leaves_a) == len(leaves_b), "pytrees have different structures"
    return max((abs(float(x) - float(y)) for la, lb in zip(leaves_a, leaves_b)
                for x, y in zip(_flat(la), _flat(lb))), default=0.0)


def _flat(leaf):
    """A leaf as a flat iterable of scalars, whether it's an array or a plain number."""
    ravel = getattr(leaf, "ravel", None)
    return ravel().tolist() if ravel is not None else [leaf]


NOISE_MARGIN = 32.0
"""Headroom over the measured floor, because one control pair is a high-variance estimate.

The measured floor is a single sample of "max over leaves of |run - run|", and that statistic
varies by a small factor between draws - measured at ~4x on this project's toy runs, which is
enough to fail a comparison against a bare single-sample floor. The margin covers that spread
without weakening the test in any way that matters: a genuinely broken resume (optimizer state
not restored, RNG key not carried, iteration count off) moves the loss by O(1e-2) or more,
which is seven-plus orders of magnitude above a noise floor of ~1e-9. Five orders of headroom
remain after this margin.
"""

ABSOLUTE_FLOOR = 1e-8
"""Minimum tolerance on a nondeterministic backend, for when a control pair happens to agree.

Two identical GPU runs CAN come out equal by luck, which would otherwise collapse the tolerance
to zero and demand bit-exactness from a backend that cannot provide it. Still six-plus orders
below any real resume bug.
"""


def tolerance_from(noise_floor: float) -> float:
    """The comparison tolerance implied by a measured noise floor on this backend."""
    if deterministic_backend():
        return 0.0
    return max(noise_floor * NOISE_MARGIN, ABSOLUTE_FLOOR)


def assert_within_noise_floor(interrupted, continuous, noise_floor: float, what: str) -> None:
    """Asserts interrupted matches continuous no worse than the backend's own run-to-run noise.

    noise_floor comes from running the continuous configuration twice; see this module's
    docstring. On a deterministic backend the tolerance is exactly 0.0, so this stays a strict
    bit-exactness test there - which is where the property is genuinely guaranteed, and where
    it is verified in CI.
    """
    if deterministic_backend():
        assert noise_floor == 0.0, (
            f"{what}: backend reports as deterministic but two identical runs differed by "
            f"{noise_floor} - the determinism contract in placax.reproducibility is wrong"
        )
    tolerance = tolerance_from(noise_floor)
    difference = max_abs_difference(interrupted, continuous)
    assert difference <= tolerance, (
        f"{what}: resuming from a checkpoint changed the run by {difference}, which exceeds "
        f"the tolerance of {tolerance} derived from this backend's own run-to-run noise floor "
        f"of {noise_floor} (measured from two identical continuous runs). A real resume bug "
        f"diverges by many orders of magnitude more than this; backend nondeterminism does not"
    )
