from placax.netlist.budget import freeze_order, truncate_to_budget  # noqa: F401  must precede jax imports
from placax.netlist.order import area_desc_order


def test_truncate_to_budget_keeps_only_the_first_n_by_order() -> None:
    macro_sizes = {"small": (1.0, 1.0), "big": (10.0, 10.0), "medium": (3.0, 3.0)}
    kept_sizes, kept_nets, kept_order = truncate_to_budget(macro_sizes, [], budget=2, order_fn=area_desc_order)
    assert set(kept_sizes) == {"big", "medium"}  # top 2 by area
    assert kept_order == ["big", "medium"]  # order preserved, not just membership


def test_truncate_to_budget_drops_nets_left_with_fewer_than_two_macros() -> None:
    macro_sizes = {"a": (1.0, 1.0), "b": (1.0, 1.0), "c": (1.0, 1.0)}
    nets = [
        [("a", 0.0, 0.0), ("c", 0.0, 0.0)],  # "c" gets dropped -> net left with 1 macro -> dropped
        [("a", 0.0, 0.0), ("b", 0.0, 0.0)],  # both kept -> net survives
    ]
    from placax.netlist.order import alphabetical_order

    _kept_sizes, kept_nets, _kept_order = truncate_to_budget(macro_sizes, nets, budget=2, order_fn=alphabetical_order)
    assert kept_nets == [[("a", 0.0, 0.0), ("b", 0.0, 0.0)]]


def test_truncate_to_budget_is_a_noop_when_budget_covers_everything() -> None:
    macro_sizes = {"a": (1.0, 1.0), "b": (2.0, 2.0)}
    nets = [[("a", 0.0, 0.0), ("b", 0.0, 0.0)]]
    kept_sizes, kept_nets, kept_order = truncate_to_budget(macro_sizes, nets, budget=10)
    assert kept_sizes == macro_sizes
    assert kept_nets == nets
    assert kept_order == sorted(macro_sizes)  # default order_fn is alphabetical_order


def test_freeze_order_ignores_its_arguments_and_always_returns_the_frozen_order() -> None:
    fixed = freeze_order(["z", "a", "m"])
    assert fixed({"anything": (1.0, 1.0)}, []) == ["z", "a", "m"]
    assert fixed({}, [[("x", 0.0, 0.0)]]) == ["z", "a", "m"]  # same result regardless of input


def test_truncate_then_freeze_order_matches_build_padded_arrays_directly() -> None:
    # The actual bug this pair of functions fixes: re-running a net-structure-sensitive order_fn
    # on the truncated netlist can rank differently than the order that was used to pick the
    # truncated set in the first place. Freezing that first order and feeding it to
    # build_padded_arrays must reproduce it exactly, not re-derive a (possibly different) one.
    from placax.netlist.order import connectivity_order
    from placax.netlist.padding import build_padded_arrays

    macro_sizes = {"Hub": (1.0, 1.0), "Mid": (1.0, 1.0), "Leaf": (1.0, 1.0), "Far": (1.0, 1.0)}
    nets = [
        [("Hub", 0.0, 0.0), ("Mid", 0.0, 0.0)],
        [("Hub", 0.0, 0.0), ("Leaf", 0.0, 0.0)],
        [("Mid", 0.0, 0.0), ("Far", 0.0, 0.0)],
    ]
    kept_sizes, kept_nets, kept_order = truncate_to_budget(macro_sizes, nets, budget=3, order_fn=connectivity_order)
    name_to_idx, *_ = build_padded_arrays(kept_sizes, kept_nets, order_fn=freeze_order(kept_order))
    assert list(name_to_idx) == kept_order
