import json
import pathlib

from placax.extras.mst import _manhattan, hpwl_wirelength, mst_wirelength, prim_mst_edges

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "mst"


def _mst_length(points: list[tuple[float, float]]) -> float:
    return sum(_manhattan(points[i], points[j]) for i, j in prim_mst_edges(points))


def test_prim_mst_edges_single_point_is_empty() -> None:
    assert prim_mst_edges([(0.0, 0.0)]) == []


def test_prim_mst_length_l_shape() -> None:
    # A at origin, B ten units right, C ten units up - MST connects
    # A-B (10) and A-C (10), star-shaped from the corner.
    points = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    edges = prim_mst_edges(points)
    assert len(edges) == 2  # a tree over 3 points has exactly 2 edges
    assert _mst_length(points) == 20.0


def test_prim_mst_length_collinear_chain() -> None:
    # Points at 0, 5, 10, 20 on a line - MST is just the chain, 20 total,
    # not e.g. connecting 0 directly to 20 and missing the middle points.
    points = [(0.0, 0.0), (10.0, 0.0), (5.0, 0.0), (20.0, 0.0)]
    assert _mst_length(points) == 20.0


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


def test_hpwl_wirelength_sums_bbox_per_net() -> None:
    positions = {"A": (0.0, 0.0), "B": (10.0, 4.0), "C": (0.0, 10.0)}
    nets = [
        [("A", 0.0, 0.0), ("B", 0.0, 0.0)],  # bbox 10 wide x 4 tall -> 14
        [("A", 0.0, 0.0), ("C", 0.0, 0.0)],  # bbox 0 wide x 10 tall -> 10
    ]
    assert hpwl_wirelength(positions, nets) == 24.0


def test_hpwl_wirelength_handles_high_fanout_nets_fast() -> None:
    # HPWL must stay O(k) per net (unlike MST's O(k^2)) - a single net with thousands of pins (a
    # clock/reset net in a real full netlist) must not be slow.
    positions = {f"c{i}": (float(i), 0.0) for i in range(5000)}
    net = [(f"c{i}", 0.0, 0.0) for i in range(5000)]
    assert hpwl_wirelength(positions, [net]) == 4999.0
