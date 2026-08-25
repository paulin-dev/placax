from placax.netlist.order import area_desc_order  # noqa: F401  must precede jax imports
from placax_agents.benchmark import Benchmark  # noqa: F401


def _write_tiny_bookshelf(tmp_path):
    (tmp_path / "sample.aux").write_text("RowBasedPlacement : sample.nodes sample.nets sample.wts sample.pl sample.scl\n")
    # Names deliberately chosen so alphabetical and area-descending order
    # disagree on what comes first - "asmall" < "zbig" alphabetically, but
    # zbig's area (100) is much larger than asmall's (1).
    (tmp_path / "sample.nodes").write_text(
        "UCLA nodes 1.0\n"
        "NumNodes : 2\n"
        "NumTerminals : 2\n"
        "asmall 1 1 terminal\n"
        "zbig 10 10 terminal\n"
    )
    (tmp_path / "sample.nets").write_text(
        "UCLA nets 1.0\n"
        "NumNets : 0\n"
        "NumPins : 0\n"
    )
    return tmp_path


def test_benchmark_load_uses_alphabetical_order_by_default(tmp_path) -> None:
    benchmark_dir = _write_tiny_bookshelf(tmp_path)
    benchmark = Benchmark.load(benchmark_dir, grid=4)
    assert benchmark.sizes_array.shape == (2, 2)
    assert benchmark.sizes_array[0].tolist() == [1.0, 1.0]  # "asmall" sorts first alphabetically


def test_benchmark_load_accepts_a_custom_order_fn(tmp_path) -> None:
    benchmark_dir = _write_tiny_bookshelf(tmp_path)
    benchmark = Benchmark.load(benchmark_dir, grid=4, order_fn=area_desc_order)
    assert benchmark.sizes_array[0].tolist() == [10.0, 10.0]  # "zbig" placed first by area_desc_order
    assert benchmark.sizes_array[1].tolist() == [1.0, 1.0]
