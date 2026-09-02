"""Plain (non-JAX) wirelength metrics for one-off/occasional reporting, given name-keyed real-unit
positions: MST-based (Prim's, Manhattan distance - more accurate than HPWL but O(k^2) per net, so only
safe on macro-scale netlists) and HPWL (O(k) per net, safe even on a full post-placement netlist with
high-fanout nets like clock/reset that would make MST prohibitively slow, or blow up memory if padded
into placax.extras.rewards.hpwl's JAX/fixed-shape-array training-hot-path form)."""
from placax.types import Nets


def _manhattan(a: tuple[float, float], b: tuple[float, float]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def prim_mst_edges(points: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """Manhattan MST edges (as point-index pairs) via Prim's algorithm, O(k^2), favoring obviously-correct
    plain arrays over speed. Reused both for MST wirelength (sum edge lengths) and for rendering (draw
    edge segments)."""
    n = len(points)
    if n < 2:
        return []

    # 1. Start the tree with the first point; min_dist[i] tracks its cheapest known distance to the tree,
    # and parent[i] the in-tree point that achieved it (i.e. the edge that would connect it).
    in_tree = [False] * n
    in_tree[0] = True
    min_dist = [_manhattan(points[0], points[i]) for i in range(n)]
    parent = [0] * n

    # 2. Repeatedly add the closest remaining point via its recorded edge, then relax the rest through it.
    edges = []
    for _ in range(n - 1):
        best_j = min((j for j in range(n) if not in_tree[j]), key=lambda j: min_dist[j])
        edges.append((parent[best_j], best_j))
        in_tree[best_j] = True
        for j in range(n):
            if not in_tree[j]:
                d = _manhattan(points[best_j], points[j])
                if d < min_dist[j]:
                    min_dist[j] = d
                    parent[j] = best_j
    return edges


def mst_wirelength(macro_positions: dict[str, tuple[float, float]], nets: Nets) -> float:
    """Sum of per-net MST wirelength, given each macro's real (x, y) center."""
    total = 0.0
    for net in nets:
        # Turn each net's (macro, offset) pins into absolute pin coordinates, then MST them.
        points = [
            (macro_positions[name][0] + x_off, macro_positions[name][1] + y_off)
            for name, x_off, y_off in net
        ]
        total += sum(_manhattan(points[i], points[j]) for i, j in prim_mst_edges(points))
    return total


def hpwl_wirelength(positions: dict[str, tuple[float, float]], nets: Nets) -> float:
    """Sum of per-net half-perimeter wirelength (bbox width + height), given each node's real (x, y)
    center. O(k) per net regardless of fanout, unlike mst_wirelength - the right metric for a full
    post-placement netlist (macros AND cells), which can include very-high-degree nets (clock, reset)."""
    total = 0.0
    for net in nets:
        xs = [positions[name][0] + x_off for name, x_off, _y_off in net]
        ys = [positions[name][1] + y_off for name, _x_off, y_off in net]
        total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total
