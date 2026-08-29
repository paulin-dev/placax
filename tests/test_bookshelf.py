import pathlib

import pytest

from placax.netlist.bookshelf import load_bookshelf, parse_nets, parse_nodes

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "bookshelf"
REAL_ADAPTEC1 = pathlib.Path("/home/claude/maskplace/maskplace/adaptec1")


def test_parse_nodes_keeps_only_terminals() -> None:
    macros = parse_nodes(FIXTURES / "sample.nodes")
    assert set(macros) == {"m1", "m2", "m3"}  # c1, c2 excluded - not terminals
    assert macros["m1"] == (100.0, 200.0)


def test_parse_nets_keeps_real_offsets() -> None:
    macros = parse_nodes(FIXTURES / "sample.nodes")
    nets = parse_nets(FIXTURES / "sample.nets", set(macros))
    # n0's m1 pin has offset (-1.0, -1.0) in the fixture - must survive parsing
    n0 = next(net for net in nets if {name for name, _x, _y in net} == {"m1", "m2"})
    m1_pin = next(pin for pin in n0 if pin[0] == "m1")
    assert m1_pin == ("m1", -1.0, -1.0)


def test_parse_nets_drops_net_whose_pins_collapse_to_one_macro() -> None:
    # n1 connects two pins of m1 to itself - only 1 distinct macro even
    # though it has 2 pins. Real bug, found against real adaptec1 data
    # (degree-1 nets leaking into output).
    macros = parse_nodes(FIXTURES / "sample.nodes")
    nets = parse_nets(FIXTURES / "sample.nets", set(macros))
    assert all(len({name for name, _x, _y in net}) >= 2 for net in nets)


def test_parse_nets_dedupes_multiple_pins_on_one_macro() -> None:
    # n2 has 2 distinct macros (m2, m3) but m3 appears twice, at offsets
    # (0.0,0.0) and (1.0,1.0) - only the FIRST offset survives, matching
    # MaskPlace's own reference loader (place_db.py's read_net_file) and
    # DREAMPlace's Bookshelf parser, both of which keep one pin per
    # (net, macro) pair. Keeping every duplicate instead inflates that net's
    # bounding box and any degree-based ordering built from it - confirmed
    # against real adaptec1 data, where ~20% of nets have a macro like this.
    macros = parse_nodes(FIXTURES / "sample.nodes")
    nets = parse_nets(FIXTURES / "sample.nets", set(macros))
    n2 = next(net for net in nets if {name for name, _x, _y in net} == {"m2", "m3"})
    assert len(n2) == 2
    m3_pins = [(x, y) for name, x, y in n2 if name == "m3"]
    assert m3_pins == [(0.0, 0.0)]


@pytest.mark.skipif(not REAL_ADAPTEC1.exists(), reason="real adaptec1 benchmark not available")
def test_load_bookshelf_matches_independent_placedb_reference() -> None:
    # Cross-check against numbers independently produced by MaskPlace's own
    # PlaceDB parser (Section 4 correctness gate) - two different parsers,
    # same real data, should converge exactly.
    macros, nets = load_bookshelf(REAL_ADAPTEC1, "adaptec1")
    assert len(macros) == 543
    assert len(nets) == 693

    # distinct-macro degree matches the original cross-check exactly
    macro_degrees = [len({name for name, _x, _y in net}) for net in nets]
    assert min(macro_degrees) == 2
    assert max(macro_degrees) == 349

    # pin count now equals macro degree everywhere - deduping to one pin per
    # (net, macro) pair means a net can never have more pins than distinct
    # macros, matching the reference parser exactly.
    pin_counts = [len(net) for net in nets]
    assert pin_counts == macro_degrees

    # real offset, not a placeholder - matches the o211430 example used
    # throughout the conversation exactly
    all_pins = [pin for net in nets for pin in net]
    assert ("o211430", -248.5, 16.0) in all_pins
