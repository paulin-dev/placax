"""Pluggable OrderFns deciding what sequence macros get placed in, swappable via build_padded_arrays(order_fn=...)."""
from placax.types import Nets, OrderFn, SizeMap


def alphabetical_order(macro_sizes: SizeMap, nets: Nets) -> list[str]:
    """Deterministic, with no opinion on placement quality - the historical default."""
    return sorted(macro_sizes)


def area_desc_order(macro_sizes: SizeMap, nets: Nets) -> list[str]:
    """Largest footprint first, a common heuristic to place the hardest-to-fit macros while the canvas is emptiest."""
    return sorted(macro_sizes, key=lambda name: -macro_sizes[name][0] * macro_sizes[name][1])


def _net_degree(nets: Nets) -> dict[str, int]:
    """How many nets each macro participates in."""
    degree: dict[str, int] = {}
    for net in nets:
        for name, _x, _y in net:
            degree[name] = degree.get(name, 0) + 1
    return degree


def _macro_nets(macro_sizes: SizeMap, nets: Nets) -> dict[str, set[int]]:
    """macro name -> set of net indices it participates in."""
    result: dict[str, set[int]] = {name: set() for name in macro_sizes}
    for net_idx, net in enumerate(nets):
        for name, _x, _y in net:
            result[name].add(net_idx)
    return result


def _adjacency(macro_sizes: SizeMap, nets: Nets) -> dict[str, set[str]]:
    """macro name -> set of OTHER macro names it shares at least one net with (unweighted)."""
    adjacency: dict[str, set[str]] = {name: set() for name in macro_sizes}
    for net in nets:
        names = {name for name, _x, _y in net}
        for name in names:
            adjacency[name] |= names - {name}
    return adjacency


def connectivity_order(
    macro_sizes: SizeMap, nets: Nets, candidate_weight: float = 1.0, degree_weight: float = 1000.0
) -> list[str]:
    """Reproduces MaskPlace's topology order: seed with the highest-degree macro, then greedily add the
    remaining macro maximizing `candidates*candidate_weight + degree*degree_weight + area`
    (candidates = count of already-placed macros it's adjacent to), ties broken by name.

    candidate_weight/degree_weight default to MaskPlace's own weights for most benchmarks; MaskPlace
    itself overrides these per-benchmark (e.g. "ariane" uses candidate_weight=30000, "bigblue3" uses
    degree_weight=100000) - pass those explicitly (e.g. via connectivity_order_for) when reproducing
    a specific benchmark's ordering."""
    if not macro_sizes:
        return []
    degree = _net_degree(nets)
    adjacency = _adjacency(macro_sizes, nets)
    area = {name: w * h for name, (w, h) in macro_sizes.items()}

    # Seed the order with the single most-connected macro (ties broken by name).
    remaining = set(macro_sizes)
    first = min(remaining, key=lambda n: (-degree.get(n, 0), n))
    order = [first]
    remaining.discard(first)
    placed = {first}

    while remaining:
        # candidates[v] = how many DISTINCT already-placed macros v is adjacent to.
        candidates: dict[str, int] = {}
        for name in placed:
            for neighbor in adjacency[name]:
                if neighbor not in placed:
                    candidates[neighbor] = candidates.get(neighbor, 0) + 1

        def sort_key(name: str) -> tuple[float, str]:
            score = candidates.get(name, 0) * candidate_weight + degree.get(name, 0) * degree_weight + area[name]
            return (-score, name)

        next_name = min(remaining, key=sort_key)
        order.append(next_name)
        remaining.discard(next_name)
        placed.add(next_name)

    return order


def connectivity_order_for(candidate_weight: float = 1.0, degree_weight: float = 1000.0) -> OrderFn:
    """Binds connectivity_order's weights so the result matches the plain OrderFn signature - pass this
    to build_padded_arrays/truncate_to_budget instead of the bare connectivity_order when you want
    specific weights (e.g. a benchmark-specific override) applied consistently."""

    def order_fn(macro_sizes: SizeMap, nets: Nets) -> list[str]:
        return connectivity_order(macro_sizes, nets, candidate_weight=candidate_weight, degree_weight=degree_weight)

    return order_fn
