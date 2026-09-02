from placax.core import reset  # noqa: F401  must precede jax imports
from placax.netlist.order import area_desc_order  # noqa: F401
from placax_agents.benchmark import Benchmark  # noqa: F401
from placax_agents.policy.scale import to_grid_units  # noqa: F401

import jax.numpy as jnp


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


def test_benchmark_load_keeps_name_to_idx_matching_sizes_array_rows(tmp_path) -> None:
    # name_to_idx must round-trip back to the exact row each macro's size/position lives at -
    # needed to turn a finished rollout's positions back into a {name: (x, y)} dict.
    benchmark_dir = _write_tiny_bookshelf(tmp_path)
    benchmark = Benchmark.load(benchmark_dir, grid=4)
    assert benchmark.name_to_idx == {"asmall": 0, "zbig": 1}
    for name, idx in benchmark.name_to_idx.items():
        assert benchmark.sizes_array[idx].tolist() == list(benchmark.macro_sizes[name])


def test_benchmark_load_accepts_a_custom_order_fn(tmp_path) -> None:
    benchmark_dir = _write_tiny_bookshelf(tmp_path)
    benchmark = Benchmark.load(benchmark_dir, grid=4, order_fn=area_desc_order)
    assert benchmark.sizes_array[0].tolist() == [10.0, 10.0]  # "zbig" placed first by area_desc_order
    assert benchmark.sizes_array[1].tolist() == [1.0, 1.0]


def test_benchmark_state_fn_binds_the_real_cell_size(tmp_path) -> None:
    # The bare `observation` default (still what evaluate()/collect_rollout() fall back to unless
    # given a state_fn) takes cell_size=1.0 unless told otherwise. benchmark.state_fn must actually
    # bind the real cell_size, not just be another equally-broken default - checked here by placing
    # one macro and confirming its rendered canvas footprint matches to_grid_units at the real
    # cell_size, which would be wrong (a differently-sized footprint) at cell_size=1.0.
    benchmark_dir = _write_tiny_bookshelf(tmp_path)
    benchmark = Benchmark.load(benchmark_dir, grid=4)
    assert benchmark.cell_size != 1.0

    positions = reset(benchmark.params).positions.at[0].set(jnp.array([0, 0]))
    state = reset(benchmark.params).replace(positions=positions, step=1)
    obs = benchmark.state_fn(state, benchmark.params, benchmark.sizes_array)
    grid_sizes = to_grid_units(benchmark.sizes_array, benchmark.cell_size)
    w0, h0 = int(grid_sizes[0, 0]), int(grid_sizes[0, 1])
    assert bool(obs["canvas"][:w0, :h0].all())  # the placed macro's full grid-unit footprint is filled
    assert not bool(obs["canvas"][w0:, h0:].any())  # nothing beyond it is


def test_benchmark_load_accepts_a_macro_budget(tmp_path) -> None:
    benchmark_dir = _write_tiny_bookshelf(tmp_path)
    benchmark = Benchmark.load(benchmark_dir, grid=4, order_fn=area_desc_order, macro_budget=1)
    assert benchmark.params.n_macros == 1
    assert benchmark.sizes_array.shape == (1, 2)
    assert benchmark.sizes_array[0].tolist() == [10.0, 10.0]  # only "zbig" (the larger one) kept
