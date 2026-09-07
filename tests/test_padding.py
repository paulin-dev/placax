import pathlib

from placax.netlist.padding import build_padded_arrays  # noqa: F401  must precede jax imports
from placax.extras.rewards import hpwl  # noqa: F401

import jax.numpy as jnp
import numpy as np
import pytest

from tests.real_benchmarks import ADAPTEC1 as REAL_ADAPTEC1


def test_build_padded_arrays_small_hand_example() -> None:
    macro_sizes = {"m1": (10.0, 20.0), "m2": (5.0, 5.0)}
    nets = [[("m1", 0.0, 0.0), ("m2", 1.0, -1.0)]]

    name_to_idx, sizes_array, padded_pin_idx, padded_pin_offset, valid_mask = build_padded_arrays(
        macro_sizes, nets
    )

    assert set(name_to_idx) == {"m1", "m2"}
    assert sizes_array.shape == (2, 2)
    assert sizes_array[name_to_idx["m1"]].tolist() == [10.0, 20.0]

    assert padded_pin_idx.shape == (1, 2)
    assert padded_pin_offset.shape == (1, 2, 2)
    assert valid_mask.shape == (1, 2)
    assert valid_mask.all()  # only one net, degree 2, no padding needed

    m1_slot = int((padded_pin_idx[0] == name_to_idx["m1"]).argmax())
    assert padded_pin_offset[0, m1_slot].tolist() == [0.0, 0.0]


def test_build_padded_arrays_pads_variable_degree_nets() -> None:
    macro_sizes = {"a": (1.0, 1.0), "b": (1.0, 1.0), "c": (1.0, 1.0)}
    nets = [
        [("a", 0.0, 0.0), ("b", 0.0, 0.0)],  # degree 2
        [("a", 0.0, 0.0), ("b", 0.0, 0.0), ("c", 0.0, 0.0)],  # degree 3
    ]
    _name_to_idx, _sizes, padded_pin_idx, _offset, valid_mask = build_padded_arrays(macro_sizes, nets)

    assert padded_pin_idx.shape == (2, 3)  # padded to the max degree (3)
    assert valid_mask[0].tolist() == [True, True, False]  # net 0 padded
    assert valid_mask[1].tolist() == [True, True, True]  # net 1 uses all 3 slots


def test_build_padded_arrays_empty_nets() -> None:
    _name_to_idx, sizes_array, padded_pin_idx, _offset, _mask = build_padded_arrays(
        {"a": (1.0, 1.0)}, []
    )
    assert sizes_array.shape == (1, 2)
    assert padded_pin_idx.shape == (0, 0)


@pytest.mark.skipif(not REAL_ADAPTEC1.exists(), reason="real adaptec1 benchmark not available")
def test_build_padded_arrays_real_end_to_end_matches_established_hpwl() -> None:
    # Full real chain: load_netlist -> build_padded_arrays -> hpwl, on the same fixed random
    # placement the golden value was established against - confirming build_padded_arrays wires
    # everything together correctly for real, not just structurally.
    #
    # Re-baselined from 517380.0 to 430309.0: the original was established before parse_nets
    # started deduping to one pin per (net, macro), and every extra duplicate pin could only widen
    # a net's bounding box, so HPWL falling is the expected direction of that change. The netlist
    # itself is still independently cross-checked against MaskPlace's own PlaceDB parser in
    # test_bookshelf (543 macros, 693 nets, degrees 2..349, exact pin offsets) - only this derived
    # constant was stale, and only because this test skipped for as long as it did.
    from tests.real_benchmarks import load_real_netlist

    macro_sizes, nets = load_real_netlist(REAL_ADAPTEC1)
    name_to_idx, sizes_array, padded_pin_idx, padded_pin_offset, valid_mask = build_padded_arrays(
        macro_sizes, nets
    )
    # 349 = the max distinct-macro degree over all nets. It was 1313 (raw pin count) before
    # parse_nets started deduping to one pin per (net, macro); this assertion stayed stale for
    # as long as the test was skipped. test_bookshelf asserts the same 349 independently.
    assert padded_pin_idx.shape == (693, 349)
    assert sizes_array.shape == (543, 2)

    grid_pos = np.load(pathlib.Path(__file__).parent / "fixtures" / "padding" / "adaptec1_random_grid_pos.npy")
    macro_names_order = np.load(
        pathlib.Path(__file__).parent / "fixtures" / "padding" / "adaptec1_macro_names_order.npy",
        allow_pickle=True,
    )
    ratio = 10.0
    name_to_gridpos = {name: grid_pos[i] for i, name in enumerate(macro_names_order)}

    positions = np.zeros((len(name_to_idx), 2), dtype=np.float32)
    for name, idx in name_to_idx.items():
        gx, gy = name_to_gridpos[name]
        w, h = sizes_array[idx]
        positions[idx] = [gx * ratio + float(w) / 2.0, gy * ratio + float(h) / 2.0]

    result = hpwl(jnp.array(positions), padded_pin_idx, padded_pin_offset, valid_mask)
    assert abs(float(result) - 430309.0) < 1e-3
