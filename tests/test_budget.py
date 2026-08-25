from placax.netlist.budget import truncate_to_budget  # noqa: F401  must precede jax imports
from placax.netlist.order import area_desc_order


def test_truncate_to_budget_keeps_only_the_first_n_by_order() -> None:
    macro_sizes = {"small": (1.0, 1.0), "big": (10.0, 10.0), "medium": (3.0, 3.0)}
    kept_sizes, kept_nets = truncate_to_budget(macro_sizes, [], budget=2, order_fn=area_desc_order)
    assert set(kept_sizes) == {"big", "medium"}  # top 2 by area


def test_truncate_to_budget_drops_nets_left_with_fewer_than_two_macros() -> None:
    macro_sizes = {"a": (1.0, 1.0), "b": (1.0, 1.0), "c": (1.0, 1.0)}
    nets = [
        [("a", 0.0, 0.0), ("c", 0.0, 0.0)],  # "c" gets dropped -> net left with 1 macro -> dropped
        [("a", 0.0, 0.0), ("b", 0.0, 0.0)],  # both kept -> net survives
    ]
    from placax.netlist.order import alphabetical_order

    _kept_sizes, kept_nets = truncate_to_budget(macro_sizes, nets, budget=2, order_fn=alphabetical_order)
    assert kept_nets == [[("a", 0.0, 0.0), ("b", 0.0, 0.0)]]


def test_truncate_to_budget_is_a_noop_when_budget_covers_everything() -> None:
    macro_sizes = {"a": (1.0, 1.0), "b": (2.0, 2.0)}
    nets = [[("a", 0.0, 0.0), ("b", 0.0, 0.0)]]
    kept_sizes, kept_nets = truncate_to_budget(macro_sizes, nets, budget=10)
    assert kept_sizes == macro_sizes
    assert kept_nets == nets
