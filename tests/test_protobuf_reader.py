import pathlib

import pytest

from placax.netlist.protobuf_reader import load_protobuf, parse_macro_pins, parse_macros, parse_nets

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "protobuf"
REAL_ARIANE = pathlib.Path("/tmp/canonical_ariane.pb.txt")


def test_parse_macros() -> None:
    macros = parse_macros((FIXTURES / "sample.pb.txt").read_text())
    assert macros == {"macroA": (29.355, 19.26), "macroB": (29.355, 19.26)}


def test_parse_macro_pins_keeps_real_offsets() -> None:
    pins = parse_macro_pins((FIXTURES / "sample.pb.txt").read_text())
    assert ("macroA", -14.3355, 5.53) in pins
    assert ("macroB", 2.5, -1.1) in pins


def test_parse_nets_finds_cross_macro_hub_only() -> None:
    # Grp_10 references pins on both macroA and macroB - a real net.
    # Grp_11 references two pins, both on macroA - not a net, filtered.
    # Grp_12 references two pins on macroA plus one on macroB - a real net (see the dedup test below).
    nets = parse_nets((FIXTURES / "sample.pb.txt").read_text())
    assert len(nets) == 2
    assert sorted(nets[0]) == [("macroA", -14.3355, 5.53), ("macroB", 2.5, -1.1)]


def test_parse_nets_dedupes_multiple_pins_on_one_macro() -> None:
    # Grp_12 references macroA/ADR[0] and macroA/ADR[1] (two physical pins on
    # macroA) plus macroB/ADR[0] - only the first macroA offset seen (ADR[0])
    # should survive, matching load_bookshelf's parse_nets convention: one
    # macro-net connection contributes one point, not one per physical pin.
    nets = parse_nets((FIXTURES / "sample.pb.txt").read_text())
    grp_12 = nets[-1]
    assert sorted(grp_12) == [("macroA", -14.3355, 5.53), ("macroB", 2.5, -1.1)]


def test_load_protobuf_matches_same_shape_as_other_loaders() -> None:
    macro_sizes, nets = load_protobuf(FIXTURES / "sample.pb.txt")
    assert isinstance(macro_sizes, dict)
    assert isinstance(nets, list)
    assert all(isinstance(n, list) for n in nets)


@pytest.mark.skipif(not REAL_ARIANE.exists(), reason="real ariane protobuf not downloaded")
def test_load_protobuf_matches_published_macro_count() -> None:
    # 133 is AlphaChip's own published macro count for this exact design -
    # independently confirmed via RTL modification count and real synthesis
    # earlier in the conversation. A third, completely different parse
    # method (this one) landing on the same number is strong validation.
    macro_sizes, nets = load_protobuf(REAL_ARIANE)
    assert len(macro_sizes) == 133
    assert len(nets) == 490

    all_pins = [pin for net in nets for pin in net]
    nonzero = sum(1 for _name, x, y in all_pins if x != 0.0 or y != 0.0)
    assert nonzero == len(all_pins)  # 100% real offsets, not placeholders
