"""Truncates a netlist to its first N macros by placement order (an OrderFn decides what "first N" means)."""
from placax.netlist.order import alphabetical_order
from placax.types import Nets, OrderFn, SizeMap


def _filter_nets_to_kept_macros(nets: Nets, kept: set[str]) -> Nets:
    """Drops pins on non-kept macros, then drops nets left with <2 distinct kept macros."""
    filtered = [[(name, x, y) for name, x, y in net if name in kept] for net in nets]
    return [net for net in filtered if len({name for name, _x, _y in net}) >= 2]


def truncate_to_budget(
    macro_sizes: SizeMap, nets: Nets, budget: int, order_fn: OrderFn = alphabetical_order
) -> tuple[SizeMap, Nets]:
    """Keeps only the first `budget` macros per order_fn(macro_sizes, nets), pruning nets accordingly.

    Order is computed once on the full netlist then sliced, not recomputed after truncation, so
    an order_fn sensitive to net structure can rank differently if re-run on the truncated result."""
    # Rank all macros once, take the prefix, then drop everything else (macros and their pins).
    kept_names = order_fn(macro_sizes, nets)[:budget]
    kept_sizes = {name: macro_sizes[name] for name in kept_names}
    kept_nets = _filter_nets_to_kept_macros(nets, set(kept_names))
    return kept_sizes, kept_nets
