"""Truncates a netlist to its first N macros by placement order (an OrderFn decides what "first N" means)."""
from placax.netlist.order import alphabetical_order
from placax.types import Nets, OrderFn, SizeMap


def _filter_nets_to_kept_macros(nets: Nets, kept: set[str]) -> Nets:
    """Drops pins on non-kept macros, then drops nets left with <2 distinct kept macros."""
    filtered = [[(name, x, y) for name, x, y in net if name in kept] for net in nets]
    return [net for net in filtered if len({name for name, _x, _y in net}) >= 2]


def freeze_order(order: list[str]) -> OrderFn:
    """Wraps an already-decided macro order as an OrderFn that ignores whatever (macro_sizes, nets)
    it's called with and always returns `order` - lets a caller hand a precomputed sequence to
    anything expecting an OrderFn (e.g. build_padded_arrays), instead of that code re-deriving its
    own order and potentially getting a different answer. See truncate_to_budget's return value."""

    def fixed(_macro_sizes: SizeMap, _nets: Nets) -> list[str]:
        return order

    return fixed


def truncate_to_budget(
    macro_sizes: SizeMap, nets: Nets, budget: int, order_fn: OrderFn = alphabetical_order
) -> tuple[SizeMap, Nets, list[str]]:
    """Keeps only the first `budget` macros per order_fn(macro_sizes, nets), pruning nets accordingly.

    Also returns that same top-`budget` order, since order_fn ran here on the FULL (pre-truncation)
    netlist - an order_fn sensitive to net structure (e.g. connectivity_order) can rank differently
    if re-run on the now-truncated netlist. A caller that goes on to build_padded_arrays() should
    wrap this returned order with freeze_order() and use that, not the original order_fn, so the
    truncated netlist's final macro sequence matches what was actually decided here rather than a
    second, independent ranking of the smaller netlist."""
    # Rank all macros once, take the prefix, then drop everything else (macros and their pins).
    kept_names = order_fn(macro_sizes, nets)[:budget]
    kept_sizes = {name: macro_sizes[name] for name in kept_names}
    kept_nets = _filter_nets_to_kept_macros(nets, set(kept_names))
    return kept_sizes, kept_nets, kept_names
