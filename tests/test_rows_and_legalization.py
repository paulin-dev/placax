"""Placement rows, the canvas they define, and the legalizer that snaps to them.

None of this existed, and the gap was invisible because it never produced an error. `.scl` was
carried around as a filename; the canvas was the die extent anchored at the origin; and macros
were exported at whatever real coordinate `grid_index * cell_size` happened to give. Measured on
adaptec1 at the point these tests were written:

    canvas   legalizer   off-row macros      max displacement
    die      none        543 / 543           -
    die      row_snap    543 -> 0            649.12 units  (3.6 grid cells)
    core     row_snap    542 -> 0              6.02 units  (0.036 grid cells)

Two findings, and they are coupled. Every macro in every result this project has produced sits off
a legal row. And legalizing on the `die` canvas is expensive - 649 units - because that canvas
puts 15% of its cells outside the core area, so the legalizer has to drag macros back in. On the
`core` canvas the same legalizer moves a macro by at most half a row pitch, which is the
theoretical floor. That is why the canvas is a hashed axis and not a silent fix.
"""
import dataclasses
import pathlib

import pytest

from placax.netlist.rows import PlacementRows, from_bookshelf, from_def  # noqa: F401  precedes jax
from placax_agents.benchmark import Benchmark
from placax_agents.experiment.budget import Budget
from placax_agents.experiment.build import build
from placax_agents.experiment.config import Spec
from placax_agents.experiment.export import write_placement
from placax_agents.experiment.presets import training

import jax.numpy as jnp

SCL = """UCLA scl 1.0
NumRows : 3

CoreRow Horizontal
  Coordinate    :   100
  Height        :   10
  Sitewidth     :    2
  Sitespacing   :    2
  SubrowOrigin  :   50\tNumSites  :  25
End
CoreRow Horizontal
  Coordinate    :   110
  Height        :   10
  Sitewidth     :    2
  Sitespacing   :    2
  SubrowOrigin  :   50\tNumSites  :  25
End
CoreRow Horizontal
  Coordinate    :   120
  Height        :   10
  Sitewidth     :    2
  Sitespacing   :    2
  SubrowOrigin  :   50\tNumSites  :  25
End
"""

NODES = ("UCLA nodes 1.0\nNumNodes : 3\nNumTerminals : 3\n"
         "a 4 4 terminal\nb 2 2 terminal\nc 2 4 terminal\n")
NETS = ("UCLA nets 1.0\nNumNets : 2\nNumPins : 4\n"
        "NetDegree : 2 n0\n\ta I : 0.0 0.0\n\tb O : 0.0 0.0\n"
        "NetDegree : 2 n1\n\tb I : 0.0 0.0\n\tc O : 0.0 0.0\n")
PL = ("UCLA pl 1.0\n\n"
      "a\t50\t100\t: N\nb\t60\t110\t: N\nc\t70\t120\t: N\nfiller\t0\t0\t: N\n")


def _design(directory: pathlib.Path) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "s.aux").write_text("RowBasedPlacement : s.nodes s.nets s.wts s.pl s.scl\n")
    (directory / "s.nodes").write_text(NODES)
    (directory / "s.nets").write_text(NETS)
    (directory / "s.pl").write_text(PL)
    (directory / "s.wts").write_text("UCLA wts 1.0\n")
    (directory / "s.scl").write_text(SCL)
    return directory


# --------------------------------------------------------------------- parsing


def test_bookshelf_rows_give_the_core_rectangle_and_its_pitch(tmp_path: pathlib.Path) -> None:
    rows = from_bookshelf(_design(tmp_path / "d") / "s.scl")
    assert (rows.x0, rows.y0) == (50.0, 100.0)
    assert (rows.x1, rows.y1) == (100.0, 130.0)  # 50 + 25*2 sites; last row top = 120 + 10
    assert rows.row_pitch == 10.0 and rows.site_width == 2.0 and rows.n_rows == 3


def test_a_missing_scl_is_no_rows_rather_than_an_error(tmp_path: pathlib.Path) -> None:
    # A real answer, not a failure: protobuf designs have none either, and a caller is expected
    # to fall back to the die extent rather than invent a core area.
    assert from_bookshelf(tmp_path / "absent.scl") is None


def test_mixed_row_heights_are_refused_rather_than_averaged(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "mixed.scl"
    path.write_text(SCL.replace("Height        :   10", "Height        :   14", 1))
    with pytest.raises(NotImplementedError, match="mixes row heights"):
        from_bookshelf(path)


def test_def_rows_are_converted_from_database_units_to_microns(tmp_path: pathlib.Path) -> None:
    # Macro geometry reaches this project from LEF in microns while DEF is in database units;
    # getting this backwards is silent and puts the core area off by a factor of 1000.
    path = tmp_path / "d.def"
    path.write_text(
        "UNITS DISTANCE MICRONS 1000 ;\n"
        "ROW ROW_0 site 50000 100000 N DO 25 BY 1 STEP 2000 0 ;\n"
        "ROW ROW_1 site 50000 110000 N DO 25 BY 1 STEP 2000 0 ;\n"
    )
    rows = from_def(path)
    assert (rows.x0, rows.y0) == (50.0, 100.0)
    assert rows.site_width == 2.0 and rows.row_pitch == 10.0


# --------------------------------------------------------------------- snapping


def _rows() -> PlacementRows:
    return PlacementRows(x0=50.0, y0=100.0, x1=100.0, y1=130.0, row_pitch=10.0,
                         site_width=2.0, n_rows=3)


def test_snap_moves_a_macro_to_the_nearest_site_and_row() -> None:
    assert _rows().snap(63.4, 113.7) == (64.0, 110.0)
    assert _rows().snap(63.4, 116.0) == (64.0, 120.0)


def test_snap_never_moves_a_macro_further_than_half_a_pitch() -> None:
    # The bound that makes legalization a rounding rather than a relocation - and the reason the
    # `core` canvas matters, since it is only true for a macro already inside the core.
    rows = _rows()
    for y in [100.0, 103.0, 107.0, 111.9, 119.0, 129.0]:
        _x, snapped = rows.snap(60.0, y)
        assert abs(snapped - y) <= rows.row_pitch / 2 + 1e-9


def test_snap_keeps_a_macro_whole_inside_the_core() -> None:
    rows = _rows()
    x, y = rows.snap(999.0, 999.0, macro_width=10.0, macro_height=10.0)
    assert rows.is_legal(x, y, 10.0, 10.0)
    assert x + 10.0 <= rows.x1 and y + 10.0 <= rows.y1


def test_a_position_already_on_a_row_is_left_alone() -> None:
    assert _rows().snap(64.0, 110.0) == (64.0, 110.0)


def test_is_legal_rejects_off_row_off_site_and_out_of_core() -> None:
    rows = _rows()
    assert rows.is_legal(64.0, 110.0)
    assert not rows.is_legal(65.0, 110.0)   # off site
    assert not rows.is_legal(64.0, 113.0)   # off row
    assert not rows.is_legal(20.0, 110.0)   # left of the core


# --------------------------------------------------------------------- the canvas axis


def test_the_die_canvas_is_unchanged_and_stays_the_default(tmp_path: pathlib.Path) -> None:
    # Every result this project has produced was `die`. It has to keep meaning what it meant, or
    # the fix silently invalidates them instead of letting a hash say so.
    directory = _design(tmp_path / "d")
    assert Benchmark.load(directory, grid=8).origin == (0.0, 0.0)
    assert training(directory).environment.benchmark.canvas == "die"


def test_the_core_canvas_is_anchored_and_scaled_to_the_placement_rows(tmp_path) -> None:
    directory = _design(tmp_path / "d")
    core = Benchmark.load(directory, grid=8, canvas="core")
    rows = core.rows
    assert core.origin == (rows.x0, rows.y0)
    assert core.cell_size == rows.span / 8


def test_the_canvas_choice_changes_the_hash(tmp_path: pathlib.Path) -> None:
    # It moves cell_size, so it moves every reward and every HPWL. Two runs that differ on it are
    # not comparable, and the hash has to say so.
    directory = _design(tmp_path / "d")
    die = training(directory, budget=Budget(iterations=1))
    core = dataclasses.replace(die, environment=dataclasses.replace(
        die.environment,
        benchmark=dataclasses.replace(die.environment.benchmark, canvas="core"),
    ))
    assert die.benchmark_hash() != core.benchmark_hash()


def test_a_core_canvas_without_rows_is_refused(tmp_path: pathlib.Path) -> None:
    # Silently falling back to the die extent would be the worst outcome: the config would claim
    # `core` while the run used `die`.
    directory = _design(tmp_path / "d")
    (directory / "s.scl").unlink()
    with pytest.raises(ValueError, match="carries none"):
        Benchmark.load(directory, grid=8, canvas="core")


def test_an_unknown_canvas_is_refused(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="unknown canvas"):
        Benchmark.load(_design(tmp_path / "d"), grid=8, canvas="whatever")


# --------------------------------------------------------------------- legalization end to end


def _built(directory: pathlib.Path, canvas: str, legalization):
    config = training(directory, budget=Budget(iterations=1))
    config = dataclasses.replace(config, environment=dataclasses.replace(
        config.environment,
        benchmark=dataclasses.replace(config.environment.benchmark, grid=8, canvas=canvas),
        legalization=legalization,
    ))
    return build(config)


def _positions(built):
    return jnp.array([[i, i] for i in range(built.benchmark.params.n_macros)])


def test_the_absence_of_a_legalizer_is_recorded_rather_than_implied(tmp_path) -> None:
    # The same reason initial_placement="empty" exists: a run that legalized nothing should say
    # so, so that the day one does, assert_comparable refuses to put them in one table.
    config = training(tmp_path, budget=Budget(iterations=1))
    assert config.environment.legalization is None
    assert config.to_dict()["environment"]["legalization"] is None
    assert "legalization" in config.environment.task_identity()


def test_a_legalizer_changes_the_task_hash(tmp_path: pathlib.Path) -> None:
    directory = _design(tmp_path / "d")
    plain = training(directory, budget=Budget(iterations=1))
    snapped = dataclasses.replace(plain, environment=dataclasses.replace(
        plain.environment, legalization=Spec("row_snap")))
    assert plain.task_hash() != snapped.task_hash()


def test_export_reports_off_row_macros_even_with_no_legalizer(tmp_path: pathlib.Path) -> None:
    # The finding that started all of this: without a legalizer, macros are off-row and nothing
    # said so. Reporting it is what makes the gap visible in a run's own output.
    built = _built(_design(tmp_path / "d"), canvas="die", legalization=None)
    exported = write_placement(built, _positions(built), tmp_path / "out")
    assert exported.legalizer is None
    assert exported.off_rows_after == exported.off_rows_before
    assert exported.max_displacement == 0.0


def test_the_legalizer_puts_every_macro_on_a_row(tmp_path: pathlib.Path) -> None:
    built = _built(_design(tmp_path / "d"), canvas="core", legalization=Spec("row_snap"))
    exported = write_placement(built, _positions(built), tmp_path / "out")
    assert exported.legalizer == "row_snap"
    assert exported.off_rows_after == 0


def test_the_core_canvas_makes_legalization_cheap(tmp_path: pathlib.Path) -> None:
    """The coupling, pinned: the same legalizer costs far more on the die canvas.

    On adaptec1 the gap is 649 units versus 6. The mechanism is that the die canvas puts cells
    outside the core, so legalizing means dragging macros back in rather than rounding them onto
    the nearest row - which is a different placement from the one the agent was scored on.
    """
    directory = _design(tmp_path / "d")
    die = write_placement(
        _built(directory, "die", Spec("row_snap")), _positions(_built(directory, "die", None)),
        tmp_path / "die",
    )
    core_built = _built(directory, "core", Spec("row_snap"))
    core = write_placement(core_built, _positions(core_built), tmp_path / "core")

    assert die.off_rows_after == core.off_rows_after == 0
    assert core.max_displacement <= core_built.benchmark.rows.row_pitch / 2 + 1e-9
    assert core.max_displacement < die.max_displacement


def test_the_origin_is_applied_when_writing_but_not_inside_the_environment(tmp_path) -> None:
    """HPWL is translation-invariant, so the origin must not perturb a reward - only a file.

    If it leaked into the environment it would change every score for no reason; if it never
    reached the file the exported macros would sit outside the core by the origin's own offset.
    """
    from placax_agents.experiment.run import score

    directory = _design(tmp_path / "d")
    built = _built(directory, canvas="core", legalization=None)
    positions = _positions(built)

    exported = write_placement(built, positions, tmp_path / "out")
    placed = (exported.path.parent / "s.pl").read_text()
    origin_x, origin_y = built.benchmark.origin
    assert origin_x > 0 and origin_y > 0
    # Every written macro sits at or beyond the core's lower-left corner.
    for line in placed.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] in built.benchmark.name_to_idx:
            assert float(parts[1]) >= origin_x and float(parts[2]) >= origin_y

    # ...and the environment itself never saw it: the arrays the reward is computed over are
    # still anchored at zero, so no jitted path pays for a constant that cannot change HPWL.
    from placax_agents.policy.scale import to_real_centers

    centers = to_real_centers(positions, built.benchmark.sizes_array, built.benchmark.cell_size)
    assert float(centers[:, 0].min()) < origin_x
    assert score(built.benchmark, positions, built.n_placed)["real_hpwl"] > 0
