"""Correctness gate (build step 4): hpwl() checked against real net data from
the adaptec1 benchmark, not just a hand-built toy example.

Fixtures were derived from MaskPlace's real adaptec1 PlaceDB (693 real nets,
degree 2 to 349) via a random node-center placement. The reference HPWL was
computed independently, in plain Python, mirroring MaskPlace's own
comp_res.py bounding-box loop structure (node centers only, zero offset -
these fixtures predate offset support, added later once elsewhere in the
codebase pin offsets are now real; this test stays a valid zero-offset
regression case). Both gave 338830.0, an exact match. See conversation
history for the generation script if these fixtures ever need regenerating."""
from pathlib import Path

import numpy as np

from placax.extras.rewards import hpwl

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE_HPWL = 338830.0


def test_hpwl_matches_independent_reference_on_real_benchmark() -> None:
    positions = np.load(FIXTURES / "adaptec1_positions.npy")
    padded_pin_idx = np.load(FIXTURES / "adaptec1_padded_pin_idx.npy")
    valid_mask = np.load(FIXTURES / "adaptec1_valid_mask.npy")
    zero_offset = np.zeros(padded_pin_idx.shape + (2,))

    assert padded_pin_idx.shape == (693, 349)  # 693 real nets, max degree 349
    result = float(hpwl(positions, padded_pin_idx, zero_offset, valid_mask))
    assert abs(result - REFERENCE_HPWL) < 1e-3
