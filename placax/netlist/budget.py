"""Truncates a netlist to its first N macros by placement order -
generalizes MaskPlace's `placed_num_macro` (RL places only its N most
important macros) into a pluggable netlist-level step, not a training-
script-only flag: any OrderFn decides what "first N" means."""
from placax.netlist.order import alphabetical_order
from placax.types import Nets, OrderFn, SizeMap


def truncate_to_budget(
    macro_sizes: SizeMap, nets: Nets, budget: int, order_fn: OrderFn = alphabetical_order
) -> tuple[SizeMap, Nets]:
    """Keeps only the first `budget` macros per order_fn(macro_sizes, nets)
    - the rest are dropped from macro_sizes and from every net's pin list
    (a net left with fewer than 2 distinct kept macros is dropped
    entirely, matching how the netlist readers themselves prune nets).
    budget >= len(macro_sizes) is a no-op. Order is computed once, on the
    full netlist, then sliced - not recomputed after truncation, matching
    MaskPlace's own `node_id_to_name[:placed_num_macro]`; for an order_fn
    whose ranking depends on net structure (e.g. connectivity_order),
    build_padded_arrays() running the same order_fn again on the
    truncated result can differ slightly from this slice, since the
    truncated netlist has fewer nets to rank by."""
    kept_names = order_fn(macro_sizes, nets)[:budget]
    kept = set(kept_names)
    kept_sizes = {name: macro_sizes[name] for name in kept_names}
    kept_nets = [[(name, x, y) for name, x, y in net if name in kept] for net in nets]
    kept_nets = [net for net in kept_nets if len({name for name, _x, _y in net}) >= 2]
    return kept_sizes, kept_nets
