"""Pluggable OrderFns deciding what sequence macros get placed in, swappable via build_padded_arrays(order_fn=...)."""
from placax.types import Nets, SizeMap


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


def connectivity_order(macro_sizes: SizeMap, nets: Nets) -> list[str]:
    """Breadth-first by connectivity, keeping macros that share nets close together in placement order."""
    if not macro_sizes:
        return []
    # Precompute how many nets each macro touches, and which net indices each macro is in.
    degree = _net_degree(nets)
    macro_nets = _macro_nets(macro_sizes, nets)

    # Seed the order with the single most-connected macro.
    remaining = set(macro_sizes)
    first = min(remaining, key=lambda n: (-degree.get(n, 0), n))
    order = [first]
    remaining.discard(first)
    frontier_nets = set(macro_nets[first])

    # Repeatedly grow the order by picking whichever remaining macro shares the most
    # nets with everything placed so far (ties broken by degree, then name), and
    # folding its nets into the "frontier" so later picks stay close to the whole group.
    while remaining:
        def sort_key(name: str) -> tuple[int, int, str]:
            shared = len(frontier_nets & macro_nets[name])
            return (-shared, -degree.get(name, 0), name)

        next_name = min(remaining, key=sort_key)
        order.append(next_name)
        remaining.discard(next_name)
        frontier_nets |= macro_nets[next_name]

    return order
