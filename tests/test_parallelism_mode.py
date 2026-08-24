import pytest

from placax._device import recommended_parallelism_mode


def test_auto_detects_from_backend() -> None:
    # this sandbox has no GPU (confirmed throughout this whole build),
    # so auto-detection should recommend sequential
    assert recommended_parallelism_mode() == "sequential"


def test_override_sequential() -> None:
    assert recommended_parallelism_mode("sequential") == "sequential"


def test_override_parallel() -> None:
    assert recommended_parallelism_mode("parallel") == "parallel"


def test_invalid_override_raises() -> None:
    with pytest.raises(ValueError, match="must be 'sequential' or 'parallel'"):
        recommended_parallelism_mode("bogus")
