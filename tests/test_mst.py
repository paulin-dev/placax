import json
import pathlib

from placax.extras.mst import _prim_mst_length, mst_wirelength

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "mst"


def test_prim_mst_length_single_point_is_zero() -> None:
    assert _prim_mst_length([(0.0, 0.0)]) == 0.0


def test_prim_mst_length_l_shape() -> None:
    # A at origin, B ten units right, C ten units up - MST connects
    # A-B (10) and A-C (10), star-shaped from the corner.
    points = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    assert _prim_mst_length(points) == 20.0


def test_prim_mst_length_collinear_chain() -> None:
    # Points at 0, 5, 10, 20 on a line - MST is just the chain, 20 total,
    # not e.g. connecting 0 directly to 20 and missing the middle points.
    points = [(0.0, 0.0), (10.0, 0.0), (5.0, 0.0), (20.0, 0.0)]
    assert _prim_mst_length(points) == 20.0


def test_mst_wirelength_sums_across_nets() -> None:
    positions = {"A": (0.0, 0.0), "B": (10.0, 0.0), "C": (0.0, 10.0)}
    nets = [
        [("A", 0.0, 0.0), ("B", 0.0, 0.0)],  # single edge, dist 10
        [("A", 0.0, 0.0), ("C", 0.0, 0.0)],  # single edge, dist 10
    ]
    assert mst_wirelength(positions, nets) == 20.0


def test_mst_wirelength_matches_maskplace_prim_real_on_real_data() -> None:
    # Cross-check against MaskPlace's own prim_real (heap-based Prim's),
    # on 50 real named nets from adaptec1, using their exact net/offset
    # data - two structurally different implementations, identical input,
    # should converge exactly.
    with open(FIXTURES / "adaptec1_50nets.json") as f:
        data = json.load(f)

    nets = [[tuple(pin) for pin in net] for net in data["nets"]]
    positions = {name: tuple(pos) for name, pos in data["positions"].items()}

    result = mst_wirelength(positions, nets)
    assert abs(result - data["expected_mst_total"]) < 1e-3
