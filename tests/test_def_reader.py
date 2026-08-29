import pathlib

from placax.netlist.def_reader import (
    load_def,
    parse_components,
    parse_nets,
    resolve_macro_sizes,
    resolve_net_pin_offsets,
)
from placax.netlist.lef import parse_lef_pin_offsets

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "def"


def test_resolve_macro_sizes_converts_type_keyed_to_instance_keyed() -> None:
    components = {"u1": ("INVX1", 0.0, 0.0), "u2": ("INVX1", 10.0, 0.0)}
    cell_sizes = {"INVX1": (0.38, 1.4)}
    macro_sizes = resolve_macro_sizes(components, cell_sizes)
    assert macro_sizes == {"u1": (0.38, 1.4), "u2": (0.38, 1.4)}


def test_resolve_macro_sizes_skips_types_missing_from_lef() -> None:
    components = {"u1": ("UNKNOWN_TYPE", 0.0, 0.0)}
    macro_sizes = resolve_macro_sizes(components, cell_sizes={})
    assert macro_sizes == {}


def test_parse_components() -> None:
    def_text = (FIXTURES / "sample.def").read_text()
    components = parse_components(def_text)
    assert components["u1"] == ("INVD1BWP240H8P57PDSVT", 100.0, 200.0)
    assert len(components) == 3


def test_parse_nets_keeps_duplicate_names_as_separate_nets() -> None:
    # "buffered_net" appears twice - a real DEF pattern (buffer insertion
    # during synthesis) that silently collapsed 85193 real nets down to
    # 261 when nets were keyed by name in a dict. Must stay two nets.
    def_text = (FIXTURES / "sample.def").read_text()
    nets = parse_nets(def_text)
    # net_a, buffered_net x2, multi_pin_net - floating_net has only 1 instance
    assert len(nets) == 4
    net_instance_sets = [{inst for inst, _port in net} for net in nets]
    assert {"u1", "u2"} in net_instance_sets


def test_parse_nets_dedupes_multiple_pins_on_one_instance() -> None:
    # "multi_pin_net" (the fixture's last NETS entry) connects u1 twice (ports
    # I1 and I2) and u2 once - only the first port seen for u1 (I1) should
    # survive, matching load_bookshelf's parse_nets convention: one macro-net
    # connection contributes one point, not one per physical pin.
    def_text = (FIXTURES / "sample.def").read_text()
    nets = parse_nets(def_text)
    multi_pin_net = nets[-1]  # net_a, buffered_net x2, then multi_pin_net, in file order
    assert sorted(multi_pin_net) == [("u1", "I1"), ("u2", "I1")]


def test_parse_nets_drops_single_instance_nets() -> None:
    def_text = (FIXTURES / "sample.def").read_text()
    nets = parse_nets(def_text)
    net_instance_sets = [{inst for inst, _port in net} for net in nets]
    # floating_net only has u3 after dropping the PIN reference - excluded
    assert {"u3"} not in net_instance_sets


def test_resolve_net_pin_offsets_uses_real_lef_geometry() -> None:
    def_text = (FIXTURES / "sample.def").read_text()
    components = parse_components(def_text)
    raw_nets = parse_nets(def_text)
    pin_offsets = parse_lef_pin_offsets(FIXTURES / "sample.lef")

    nets = resolve_net_pin_offsets(raw_nets, components, pin_offsets)
    all_pins = [pin for net in nets for pin in net]
    # pin I1: RECT 0.06 0.525 0.165 0.7 -> center (0.1125, 0.6125)
    # macro SIZE 0.38 BY 1.4 -> half-size (0.19, 0.7)
    # offset = (0.1125-0.19, 0.6125-0.7) = (-0.0775, -0.0875)
    i1_pins = [p for p in all_pins if p[0] in ("u1", "u2") and abs(p[1] - -0.0775) < 1e-6]
    assert i1_pins, f"expected an I1 pin with offset -0.0775, got {all_pins}"


def test_load_def_resolves_to_same_shape_as_bookshelf() -> None:
    macro_sizes, nets = load_def(FIXTURES / "sample.def", [FIXTURES / "sample.lef"])
    assert len(macro_sizes) == 3
    assert len(nets) == 4
    # instance name maps straight to size now, matching load_bookshelf exactly -
    # no separate cell_sizes-by-type lookup needed by callers
    assert macro_sizes["u1"] == (0.38, 1.4)
    # nets carry real offsets now, not just bare instance names
    all_pins = [pin for net in nets for pin in net]
    assert any(abs(x) > 1e-6 or abs(y) > 1e-6 for _name, x, y in all_pins)
