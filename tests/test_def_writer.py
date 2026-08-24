import pathlib

import pytest

from placax.netlist.def_reader import parse_components
from placax.netlist.def_writer import write_placed_def

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "def"
REAL_ARIANE_DEF = pathlib.Path("/tmp/tilos_check/CodeElements/FormatTranslators/test/CTAriane/ariane.def")


def test_write_placed_def_updates_only_specified_components() -> None:
    original = (FIXTURES / "sample.def").read_text()
    new_positions = {"u1": (999, 888), "u2": (111, 222)}

    new_def = write_placed_def(original, new_positions)
    components = parse_components(new_def)

    assert components["u1"] == ("INVD1BWP240H8P57PDSVT", 999.0, 888.0)
    assert components["u2"] == ("INVD1BWP240H8P57PDSVT", 111.0, 222.0)
    assert components["u3"] == ("INVD1BWP240H8P57PDSVT", 500.0, 200.0)  # untouched


def test_write_placed_def_preserves_non_component_structure() -> None:
    original = (FIXTURES / "sample.def").read_text()
    new_def = write_placed_def(original, {"u1": (0, 0)})

    assert "NETS 4 ;" in new_def
    assert "DIEAREA" in new_def
    assert "DESIGN sample ;" in new_def


@pytest.mark.skipif(not REAL_ARIANE_DEF.exists(), reason="real ariane.def not available")
def test_write_placed_def_at_real_scale() -> None:
    original = REAL_ARIANE_DEF.read_text()
    components = parse_components(original)
    new_positions = {name: (i * 10, i * 20) for i, name in enumerate(components)}

    new_def = write_placed_def(original, new_positions)
    new_components = parse_components(new_def)

    assert len(new_components) == len(components)
    sample_name = list(components)[100]
    assert new_components[sample_name][1:] == (1000.0, 2000.0)
    # real floorplan structure (ROW statements) must survive untouched -
    # a downstream cell placer needs this, placax itself never reads it
    assert original.count("\nROW") == new_def.count("\nROW")
