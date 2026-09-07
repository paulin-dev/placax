import pytest

from placax._device import recommended_parallelism_mode

import jax


def test_auto_detects_from_backend() -> None:
    # Assert the RULE (CPU -> sequential, accelerator -> parallel), not one machine's answer.
    # This used to assert "sequential" unconditionally with the comment "this sandbox has no
    # GPU", which made it a test of the developer's hardware rather than of the function, and
    # it failed on every GPU host.
    expected = "sequential" if jax.default_backend() == "cpu" else "parallel"
    assert recommended_parallelism_mode() == expected


def test_override_sequential() -> None:
    assert recommended_parallelism_mode("sequential") == "sequential"


def test_override_parallel() -> None:
    assert recommended_parallelism_mode("parallel") == "parallel"


def test_invalid_override_raises() -> None:
    with pytest.raises(ValueError, match="must be 'sequential' or 'parallel'"):
        recommended_parallelism_mode("bogus")
