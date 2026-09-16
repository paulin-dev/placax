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


def test_the_last_pin_of_a_macro_is_not_dropped() -> None:
    """A silently lost pin is a net that measures shorter than it is.

    The MACRO block pattern consumed the newline after its final `END <pin>`, and the PIN pattern
    needed one - so every macro in every LEF lost its LAST pin, including the fixture below, whose
    output pin simply never appeared. Nothing raised; the design just had less connectivity than
    the file described, and every HPWL taken from a DEF design was measured against it.
    """
    import pathlib as _pathlib

    from placax.netlist.lef import parse_lef_pin_offsets

    lef = _pathlib.Path(__file__).parent / "fixtures" / "def" / "sample.lef"
    declared = {line.split()[1] for line in lef.read_text().splitlines()
                if line.strip().startswith("PIN ")}
    parsed = parse_lef_pin_offsets(lef)["INVD1BWP240H8P57PDSVT"]
    assert set(parsed) == declared == {"I1", "O1"}


def test_a_lef_whose_last_macro_ends_at_eof_still_parses(tmp_path) -> None:
    # The same missing terminator, one level up: a file with no trailing newline used to lose its
    # final MACRO entirely.
    from placax.netlist.lef import parse_lef_sizes

    lef = tmp_path / "t.lef"
    lef.write_text("MACRO A\n  SIZE 2 BY 3 ;\nEND A")   # no trailing newline
    assert parse_lef_sizes(lef) == {"A": (2.0, 3.0)}


# ------------------------------------------------------------------ a fresh floorplan

FLOORPLAN_DEF = """VERSION 5.8 ;
DESIGN fp ;
UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 100000 100000 ) ;
COMPONENTS 4 ;
    - ram0 RAM ;
    - ram1 RAM + SOURCE TIMING + UNPLACED ;
    - inv0 INV + UNPLACED ;
    - inv1 INV + PLACED ( 1000 1000 ) N ;
END COMPONENTS
NETS 3 ;
    - a ( ram0 Q ) ( inv0 A ) ( ram1 D ) + USE SIGNAL ;
    - b ( ram0 Q ) ( inv0 A ) + USE SIGNAL ;
    - c ( inv0 Y ) ( inv1 A ) + USE SIGNAL ;
END NETS
END DESIGN
"""

FLOORPLAN_LEF = """MACRO RAM
  CLASS BLOCK ;
  SIZE 20 BY 10 ;
  PIN Q
    PORT
      LAYER metal2 ;
        RECT 0 0 1 1 ;
    END
  END Q
  PIN D
    PORT
      LAYER metal2 ;
        RECT 19 9 20 10 ;
    END
  END D
END RAM
MACRO INV
  CLASS CORE ;
  SIZE 1 BY 2 ;
  PIN A
    PORT
      LAYER metal1 ;
        RECT 0 0 0.5 0.5 ;
    END
  END A
  PIN Y
    PORT
      LAYER metal1 ;
        RECT 0.5 1.5 1 2 ;
    END
  END Y
END INV
"""


def test_a_floorplan_with_nothing_placed_loads_its_blocks_as_the_macros(tmp_path) -> None:
    # Everything straight out of a floorplanner is UNPLACED. The macros are the BLOCK masters,
    # and the nets are what connects them - a standard cell is neither.
    (tmp_path / "fp.def").write_text(FLOORPLAN_DEF)
    (tmp_path / "cells.lef").write_text(FLOORPLAN_LEF)
    macro_sizes, nets = load_def(tmp_path / "fp.def", [tmp_path / "cells.lef"])
    assert macro_sizes == {"ram0": (20.0, 10.0), "ram1": (20.0, 10.0)}
    assert len(nets) == 1   # only net a joins two macros
    assert sorted(inst for inst, _x, _y in nets[0]) == ["ram0", "ram1"]
    pins = {inst: (x, y) for inst, x, y in nets[0]}
    assert pins["ram0"] == (-9.5, -4.5) and pins["ram1"] == (9.5, 4.5)


def test_lef_classes_are_read_per_master(tmp_path) -> None:
    from placax.netlist.lef import parse_lef_classes

    (tmp_path / "cells.lef").write_text(FLOORPLAN_LEF)
    assert parse_lef_classes(tmp_path / "cells.lef") == {"RAM": "BLOCK", "INV": "CORE"}


def test_a_net_wrapped_over_several_lines_is_still_one_net() -> None:
    # OpenROAD wraps high-fanout nets; a line-based reader dropped every one of them.
    text = (
        "NETS 3 ;\n"
        "    - wide ( u1 A ) ( u2 A )\n      ( u3 A ) ( u4 A ) + USE SIGNAL ;\n"
        "    - clk ( u1 CK ) ( u2 CK ) + USE CLOCK ;\n"
        "    - plain ( u3 Y ) ( u4 A ) ;\n"
        "END NETS\n"
    )
    nets = parse_nets("\n" + text)
    assert [sorted(inst for inst, _ in net) for net in nets] == [
        ["u1", "u2", "u3", "u4"], ["u3", "u4"],
    ]


def test_a_placed_designs_io_pins_count_toward_its_wirelength(tmp_path) -> None:
    from placax.netlist.def_reader import load_placed_design, parse_pins

    text = FLOORPLAN_DEF.replace(
        "END COMPONENTS\n",
        "END COMPONENTS\nPINS 1 ;\n    - clk + NET a + DIRECTION INPUT + USE SIGNAL\n"
        "      + PORT\n        + LAYER metal5 ( -100 -100 ) ( 100 100 )\n"
        "        + PLACED ( 0 50000 ) N ;\nEND PINS\n",
    ).replace("( ram0 Q ) ( inv0 A )", "( PIN clk ) ( ram0 Q ) ( inv0 A )").replace(
        "- ram0 RAM ;", "- ram0 RAM + FIXED ( 10000 10000 ) N ;")
    assert parse_pins(text) == {"clk": (0.0, 50000.0)}
    (tmp_path / "fp.def").write_text(text)
    (tmp_path / "cells.lef").write_text(FLOORPLAN_LEF)
    positions, sizes, nets = load_placed_design(tmp_path / "fp.def", [tmp_path / "cells.lef"])
    assert positions["PIN:clk"] == (0.0, 50.0) and sizes["PIN:clk"] == (0.0, 0.0)
    assert any("PIN:clk" in {inst for inst, _x, _y in net} for net in nets)
