"""MST-based wirelength (Prim's, Manhattan distance) - more accurate
than HPWL for high-fanout nets but far more expensive. Deliberately
plain Python, not jit-compiled: like MaskPlace's prim_real, it's only
for periodic reporting, never inside the training loop."""
from placax.types import Nets


def _prim_mst_length(points: list[tuple[float, float]]) -> float:
    """Manhattan MST length via Prim's algorithm, O(k^2), plain arrays -
    obviously-correct over fast."""
    n = len(points)
    if n < 2:
        return 0.0

    in_tree = [False] * n
    in_tree[0] = True
    min_dist = [
        abs(points[0][0] - points[i][0]) + abs(points[0][1] - points[i][1]) for i in range(n)
    ]

    total = 0.0
    for _ in range(n - 1):
        best_j = min((j for j in range(n) if not in_tree[j]), key=lambda j: min_dist[j])
        total += min_dist[best_j]
        in_tree[best_j] = True
        for j in range(n):
            if not in_tree[j]:
                d = abs(points[best_j][0] - points[j][0]) + abs(points[best_j][1] - points[j][1])
                min_dist[j] = min(min_dist[j], d)
    return total


def mst_wirelength(macro_positions: dict[str, tuple[float, float]], nets: Nets) -> float:
    """Sum of per-net MST wirelength. macro_positions maps macro name to
    its real (x, y) center; pin position = center + offset."""
    total = 0.0
    for net in nets:
        points = [
            (macro_positions[name][0] + x_off, macro_positions[name][1] + y_off)
            for name, x_off, y_off in net
        ]
        total += _prim_mst_length(points)
    return total
