from placax.netlist.order import alphabetical_order, area_desc_order, connectivity_order  # noqa: F401
from placax.netlist.padding import build_padded_arrays  # noqa: F401  must precede jax imports


def test_alphabetical_order_matches_sorted() -> None:
    macro_sizes = {"c": (1.0, 1.0), "a": (1.0, 1.0), "b": (1.0, 1.0)}
    assert alphabetical_order(macro_sizes, []) == ["a", "b", "c"]


def test_area_desc_order_largest_first() -> None:
    macro_sizes = {"small": (1.0, 1.0), "big": (10.0, 10.0), "medium": (3.0, 3.0)}
    assert area_desc_order(macro_sizes, []) == ["big", "medium", "small"]


def test_connectivity_order_starts_from_highest_degree_and_stays_local() -> None:
    # Star topology: "hub" touches 3 nets, each leaf touches 1 - hub must
    # come first, and every leaf must directly follow something it shares
    # a net with (trivially true here, but confirms the frontier grows).
    macro_sizes = {"hub": (1.0, 1.0), "leaf1": (1.0, 1.0), "leaf2": (1.0, 1.0), "leaf3": (1.0, 1.0)}
    nets = [
        [("hub", 0.0, 0.0), ("leaf1", 0.0, 0.0)],
        [("hub", 0.0, 0.0), ("leaf2", 0.0, 0.0)],
        [("hub", 0.0, 0.0), ("leaf3", 0.0, 0.0)],
    ]
    order = connectivity_order(macro_sizes, nets)
    assert order[0] == "hub"
    assert set(order) == set(macro_sizes)


def test_connectivity_order_handles_disconnected_macros() -> None:
    macro_sizes = {"a": (1.0, 1.0), "b": (1.0, 1.0), "isolated": (1.0, 1.0)}
    nets = [[("a", 0.0, 0.0), ("b", 0.0, 0.0)]]
    order = connectivity_order(macro_sizes, nets)
    assert set(order) == set(macro_sizes)  # every macro still appears exactly once


def test_build_padded_arrays_accepts_a_custom_order_fn() -> None:
    macro_sizes = {"c": (1.0, 1.0), "a": (10.0, 10.0), "b": (2.0, 2.0)}
    name_to_idx, _sizes, _idx, _off, _valid = build_padded_arrays(
        macro_sizes, [], order_fn=area_desc_order
    )
    assert name_to_idx == {"a": 0, "b": 1, "c": 2}  # by area: 100, 4, 1


def test_build_padded_arrays_default_order_is_alphabetical() -> None:
    macro_sizes = {"z": (1.0, 1.0), "a": (1.0, 1.0)}
    name_to_idx, _sizes, _idx, _off, _valid = build_padded_arrays(macro_sizes, [])
    assert name_to_idx == {"a": 0, "z": 1}
