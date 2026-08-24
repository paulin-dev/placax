"""MST-based wirelength (Prim's algorithm, Manhattan distance) - a more
accurate but much more expensive proxy than HPWL, particularly for
high-fanout nets where a bounding box underestimates true routed length.

Deliberately plain Python, not JAX/jit-compiled: this mirrors how
MaskPlace's own prim_real is actually used - only for periodic
reporting, never inside the training loop. Making this jit/vmap-fast
would solve a harder problem than the one that exists: Prim's is an
inherently sequential algorithm, and even a fixed-iteration version
would cost O(max_pins^2) per net - genuinely expensive for real
high-fanout nets."""
from placax.types import Nets, SizeMap


def _prim_mst_length(points: list[tuple[float, float]]) -> float:
    """Manhattan-distance MST length via Prim's algorithm, O(k^2) for k
    points. Plain array-based (not a heap) - prioritizes obviously-correct
    over fast, matching this module's occasional-use purpose."""
    n = len(points)
    if n < 2:
        return 0.0

    in_tree = [False] * n
    in_tree[0] = True
    min_dist = [abs(points[0][0] - points[i][0]) + abs(points[0][1] - points[i][1]) for i in range(n)]

    total = 0.0
    for _ in range(n - 1):
        best_j = min((j for j in range(n) if not in_tree[j]), key=lambda j: min_dist[j])
        total += min_dist[best_j]
        in_tree[best_j] = True
        for j in range(n):
            if not in_tree[j]:
                d = abs(points[best_j][0] - points[j][0]) + abs(points[best_j][1] - points[j][1])
                if d < min_dist[j]:
                    min_dist[j] = d
    return total


def mst_wirelength(macro_positions: dict[str, tuple[float, float]], nets: Nets) -> float:
    """Sum of per-net MST wirelength. macro_positions maps macro name to
    its real (x, y) center - same convention as everywhere else in
    placax, pin position = macro center + offset."""
    total = 0.0
    for net in nets:
        points = [
            (macro_positions[name][0] + x_off, macro_positions[name][1] + y_off)
            for name, x_off, y_off in net
        ]
        total += _prim_mst_length(points)
    return total
