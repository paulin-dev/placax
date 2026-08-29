import pathlib

import pytest

from placax.netlist import NetlistFormat, detect_format, load_netlist

DEF_FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "def"


def test_detects_bookshelf(tmp_path: pathlib.Path) -> None:
    (tmp_path / "adaptec1.aux").write_text("RowBasedPlacement : adaptec1.nodes")
    assert detect_format(tmp_path) is NetlistFormat.BOOKSHELF


def test_detects_def(tmp_path: pathlib.Path) -> None:
    (tmp_path / "ariane.def").write_text("VERSION 5.8 ;")
    assert detect_format(tmp_path) is NetlistFormat.DEF


def test_detects_protobuf(tmp_path: pathlib.Path) -> None:
    (tmp_path / "netlist.pb.txt").write_text("node {}")
    assert detect_format(tmp_path) is NetlistFormat.PROTOBUF


def test_detects_unconverted_verilog(tmp_path: pathlib.Path) -> None:
    (tmp_path / "ariane.sv2v.v").write_text("module ariane; endmodule")
    assert detect_format(tmp_path) is NetlistFormat.VERILOG_RTL


def test_detects_unknown(tmp_path: pathlib.Path) -> None:
    assert detect_format(tmp_path) is NetlistFormat.UNKNOWN
    with pytest.raises(ValueError, match="no recognizable netlist format"):
        load_netlist(tmp_path)


def test_aux_takes_priority_over_stray_verilog_files(tmp_path: pathlib.Path) -> None:
    (tmp_path / "adaptec1.aux").write_text("RowBasedPlacement : adaptec1.nodes")
    (tmp_path / "notes.v").write_text("not actually verilog")
    assert detect_format(tmp_path) is NetlistFormat.BOOKSHELF


def test_load_netlist_routes_def_to_real_loader() -> None:
    macro_sizes, nets = load_netlist(DEF_FIXTURES)
    assert len(macro_sizes) == 3
    assert len(nets) == 4  # net_a, buffered_net x2, multi_pin_net - floating_net has only 1 instance


def test_load_netlist_routes_protobuf_to_real_loader() -> None:
    protobuf_fixtures = pathlib.Path(__file__).parent / "fixtures" / "protobuf"
    macro_sizes, nets = load_netlist(protobuf_fixtures)
    assert len(macro_sizes) == 2
    assert len(nets) == 2  # Grp_10 and Grp_12 - Grp_11 has only 1 distinct macro


def test_load_netlist_routes_bookshelf_to_real_loader(tmp_path: pathlib.Path) -> None:
    bookshelf_fixtures = pathlib.Path(__file__).parent / "fixtures" / "bookshelf"
    (tmp_path / "sample.aux").write_text("RowBasedPlacement : sample.nodes sample.nets")
    (tmp_path / "sample.nodes").write_text((bookshelf_fixtures / "sample.nodes").read_text())
    (tmp_path / "sample.nets").write_text((bookshelf_fixtures / "sample.nets").read_text())

    macros, nets = load_netlist(tmp_path)
    assert len(macros) == 3
    assert len(nets) == 2


def test_load_netlist_raises_clearly_for_unconverted_verilog(tmp_path: pathlib.Path) -> None:
    (tmp_path / "ariane.sv2v.v").write_text("module ariane; endmodule")
    with pytest.raises(NotImplementedError, match="Synthesize it first"):
        load_netlist(tmp_path)


def test_load_netlist_returns_identical_shape_for_both_formats(tmp_path: pathlib.Path) -> None:
    # Bookshelf
    bookshelf_fixtures = pathlib.Path(__file__).parent / "fixtures" / "bookshelf"
    (tmp_path / "sample.aux").write_text("RowBasedPlacement : sample.nodes sample.nets")
    (tmp_path / "sample.nodes").write_text((bookshelf_fixtures / "sample.nodes").read_text())
    (tmp_path / "sample.nets").write_text((bookshelf_fixtures / "sample.nets").read_text())
    bookshelf_result = load_netlist(tmp_path)

    # DEF
    def_result = load_netlist(pathlib.Path(__file__).parent / "fixtures" / "def")

    for result in (bookshelf_result, def_result):
        macro_sizes, nets = result  # unpacking itself proves both are 2-tuples
        assert isinstance(macro_sizes, dict)
        assert all(isinstance(v, tuple) and len(v) == 2 for v in macro_sizes.values())
        assert isinstance(nets, list)
        assert all(isinstance(n, list) for n in nets)
