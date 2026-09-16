"""The physical flow, driven by an ExperimentConfig rather than by a script's CLI flags.

The cell placer and validator were always substitutable - `placax_tools.pipeline.place_and_validate`
names neither DREAMPlace nor OpenROAD. What they were not was *recorded*: a PPA number came out of
whichever flags `scripts/validate_design.py` happened to be given, and nothing tied it back to the
run whose placement it measured. So the last box of the architecture sat outside the
reproducibility envelope that the rest of the project is built around.

This closes it. The tools are named in `EnvironmentSpec.physical`, so they are part of the
environment hash and appear in every manifest; `evaluate_physical` constructs them from that
config plus this host's binary paths; and the result written here carries the run's own
`full_hash`, so a `ppa.json` is attributable to the exact configuration that produced the
placement it measured.

The split between the two is deliberate. WHICH tools ran changes the result and is hashed; WHERE
they are installed does not, and is not - two labs running one experiment on one design have to
compare as comparable, which an install path in the environment hash would prevent.

`evaluate_placement` is the entry point a run should use: it exports the agent's own placement
into the design's format (`experiment.export`) and measures that. Until that existed, this module
could only be pointed at a DEF someone had produced by hand, because nothing in the repository
converted a placement into one - the flow was configured and hashed but structurally unreachable
from a run.

**Driven end to end through a real OpenROAD** (the pinned ORFS image - see
`placax_tools/openroad/docker.py`). On a real sky130 design the validator's full-design HPWL agrees
with OpenROAD's own detailed placer to 0.05 um; on a converted Bookshelf design it agrees with a
hand calculation exactly. Bookshelf benchmarks reach it through `netlist/def_export.py`, which
converts the whole netlist. What a converted design can NOT give is a routed wirelength or a
timing number: its derived technology has one layer and no timing library, so those come back None
with the tool's reason in `notes` - a Bookshelf benchmark carries no process to measure against.
"""
import dataclasses
import json
import pathlib
from dataclasses import asdict, dataclass

from placax.netlist import NetlistFormat
from placax_agents.experiment.build import BuiltExperiment, build_physical
from placax_agents.experiment.export import write_placement
from placax_tools.pipeline import place_and_validate, validate_only

PPA_NAME = "ppa.json"


@dataclass(frozen=True)
class PhysicalResult:
    """A PPA measurement plus the identity of the run and the tools that produced it."""

    full_hash: str
    cell_placer: str | None
    validator: str
    def_path: str
    design_area: float | None
    utilization_pct: float | None
    timing_slack: float | None
    """None means not computed - the validator was given no liberty file or clock period. Never
    a plausible-looking substitute for a number nobody measured."""

    routed_wirelength: float | None = None
    via_count: int | None = None
    drc_violations: int | None = None
    """Present only when the validator was configured to route. Routed wirelength is the number
    every HPWL in this project is a proxy FOR, and DRC is where a placement with excellent
    wirelength is found to be unroutable - so a PPA record that cannot carry them cannot answer
    the question the proxy was standing in for."""

    hpwl: float | None = None
    """Full-design half-perimeter wirelength in microns, measured by the validator - every signal
    net, standard cells included. What `real_hpwl` in a training log is a macro-only proxy for."""

    placement_legal: bool | None = None
    """The validator's own placement checker's verdict on the measured design."""

    placement_violations: dict[str, int] = dataclasses.field(default_factory=dict)
    """The checker's failed rules with their counts - what `placement_legal=False` was about."""

    legalized: bool | None = None
    hpwl_before_legalization: float | None = None
    legalization_max_displacement: float | None = None
    legalization_mean_displacement: float | None = None
    """The validator's own final legalization (OpenROAD's detailed placement), and what it cost.
    `hpwl` is measured after it."""

    total_negative_slack: float | None = None
    tool_version: str | None = None
    """The validator's self-reported version. Numbers change between releases, so a PPA figure
    without this cannot be reproduced."""

    notes: list[str] = dataclasses.field(default_factory=list)
    """Measurements that were asked for and could not run, each with the tool's reason."""

    legalizer: str | None = None
    max_displacement: float = 0.0
    off_rows_after: int = 0
    """What it cost to make this placement physically realizable before measuring it. A PPA number
    for a placement that had to move 649 units to reach a legal row describes a different
    placement from the one the agent was scored on, and the record has to show that."""

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_physical(
    built: BuiltExperiment,
    def_path: pathlib.Path,
    lef_paths: list[pathlib.Path],
    output_dir: pathlib.Path,
    skip_cell_placement: bool = False,
    machine: dict | None = None,
) -> PhysicalResult:
    """Runs this run's configured physical flow on a macro-placed DEF and records the result.

    `skip_cell_placement` validates the DEF as given - for a design whose standard cells are
    already placed, or a Bookshelf flow where DREAMPlace has already done that natively.

    `machine` carries this host's tool locations (dreamplace_root, use_docker, gpu, the openroad
    binary). They are deliberately not part of the config: two labs running one experiment on one
    design must compare as comparable, and an install path in the environment hash would make
    that impossible. Tools already set on `built` are used as-is, which is how a caller injects
    one directly.
    """
    config = built.config
    physical = config.environment.physical
    if physical.validator is None:
        raise ValueError(
            f"experiment {config.name!r} declares no validator, so there is no physical flow to "
            f"run. Add one to EnvironmentSpec.physical (e.g. Spec('openroad')) - naming it in "
            f"the config is what makes the resulting PPA number attributable."
        )
    if not skip_cell_placement and physical.cell_placer is None:
        raise ValueError(
            f"experiment {config.name!r} declares no cell placer. Add one to "
            f"EnvironmentSpec.physical, or pass skip_cell_placement=True if this DEF's standard "
            f"cells are already placed."
        )

    # Construct whatever the caller didn't already provide, from the config plus this machine.
    cell_placer, validator = built.cell_placer, built.validator
    if cell_placer is None or validator is None:
        resolved_placer, resolved_validator = build_physical(config, machine)
        cell_placer = cell_placer or resolved_placer
        validator = validator or resolved_validator

    output_dir.mkdir(parents=True, exist_ok=True)
    if skip_cell_placement:
        ppa = validate_only(def_path, lef_paths, output_dir, validator)
        placed_def = def_path
    else:
        placed = place_and_validate(def_path, lef_paths, output_dir, cell_placer, validator)
        ppa, placed_def = placed.ppa, placed.def_path

    return _physical_result(
        config, ppa, placed_def,
        None if skip_cell_placement else physical.cell_placer.name,
    )


def _physical_result(config, ppa, placed_def, cell_placer_name) -> PhysicalResult:
    """A validator's PPAResult as this run's record."""
    return PhysicalResult(
        full_hash=config.full_hash(),
        cell_placer=cell_placer_name,
        validator=config.environment.physical.validator.name,
        def_path=str(placed_def),
        design_area=ppa.design_area,
        utilization_pct=ppa.utilization_pct,
        timing_slack=ppa.timing_slack,
        routed_wirelength=ppa.routed_wirelength,
        via_count=ppa.via_count,
        drc_violations=ppa.drc_violations,
        hpwl=ppa.hpwl,
        placement_legal=ppa.placement_legal,
        placement_violations=dict(ppa.placement_violations),
        legalized=ppa.legalized,
        hpwl_before_legalization=ppa.hpwl_before_legalization,
        legalization_max_displacement=ppa.legalization_max_displacement,
        legalization_mean_displacement=ppa.legalization_mean_displacement,
        total_negative_slack=ppa.total_negative_slack,
        tool_version=ppa.tool_version,
        notes=list(ppa.notes),
    )


def evaluate_placement(
    built: BuiltExperiment,
    positions,
    lef_paths: list[pathlib.Path],
    output_dir: pathlib.Path,
    skip_cell_placement: bool = False,
    machine: dict | None = None,
    orientations=None,
) -> PhysicalResult:
    """Measures a run's OWN placement: export it to the design's format, then run the flow on it.

    This is the link that was missing. `evaluate_physical` below has always taken a `def_path`
    "with every macro already placed", and nothing in this repository produced one - so the
    physical flow could only ever be pointed at a file someone made by hand, and no agent's
    placement could reach it. `positions` here is the array the agent handed the runner and
    `score()` measured, so the design that gets validated is the placement that was reported.

    For a Bookshelf design the LEFs are an OUTPUT of the export rather than an input: Bookshelf
    carries no cell library, so one is derived from the netlist's own geometry and appended to
    whatever `lef_paths` the caller supplied.
    """
    # Bookshelf designs are CONVERTED rather than refused. This used to raise, which made the
    # whole physical box unreachable from any benchmark that ships: OpenROAD reads DEF/LEF, every
    # benchmark here is Bookshelf or protobuf, and nothing bridged them. The conversion exports
    # the full netlist - macros FIXED where the agent put them, standard cells UNPLACED for the
    # cell placer - and writes the cell library it derived, which the validator then needs.
    exported = write_placement(
        built, positions, output_dir / "placement", orientations, as_def=True
    )
    if exported.format is not NetlistFormat.DEF:  # noqa: SIM102  - the message needs `exported`
        raise NotImplementedError(
            f"this run's design is {exported.format.value}, and the validator reads DEF/LEF only. "
            f"Bookshelf designs are converted (placax/netlist/def_export.py); a protobuf netlist "
            f"carries no cell geometry to convert, so bring a DEF/LEF or a Bookshelf design here."
        )
    result = evaluate_physical(
        built, exported.path, list(lef_paths) + list(exported.lef_paths), output_dir,
        skip_cell_placement=skip_cell_placement, machine=machine,
    )
    # Carry what legalization cost into the PPA record: this number describes the design that was
    # actually measured, which is the legalized one, not the raw grid placement.
    return dataclasses.replace(
        result, legalizer=exported.legalizer, max_displacement=exported.max_displacement,
        off_rows_after=exported.off_rows_after,
    )


def write_ppa(output_dir: pathlib.Path, result: PhysicalResult) -> pathlib.Path:
    """Writes ppa.json next to the run's manifest, carrying the run hash it belongs to."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / PPA_NAME
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


@dataclass(frozen=True)
class FullDesign:
    """A Bookshelf design with every node placed, and what was measured on it."""

    result_pl: pathlib.Path
    """The cell placer's output: every macro and every standard cell."""

    positions: dict
    """{node: (x, y)} lower-left corners, every node the design has."""

    sizes: dict
    """{node: (width, height)}, every node."""

    macro_names: frozenset
    """The nodes the agent placed - the rest are the cell placer's."""

    full_hpwl: float
    """Half-perimeter wirelength over every net, from this project's own computation."""

    ppa: PhysicalResult | None
    """The validator's measurement, or None when the config names no validator."""


def place_cells_and_measure(
    built: BuiltExperiment,
    positions,
    output_dir: pathlib.Path,
    machine: dict | None = None,
    orientations=None,
) -> FullDesign:
    """An agent's macro placement, finished and measured: the configured physical stack on it.

    What `scripts/run_pipeline.py` does after its rollout, as a function, so that a comparison
    can put every agent through the identical flow. Bookshelf only, because DREAMPlace's
    Bookshelf mode is the reliable one for these designs:

      1. the macros are written as a `.pl`/`.aux` (the configured legalizer applies),
      2. the configured cell placer places every standard cell around them,
      3. the full-design HPWL is computed here, independently of any tool,
      4. with a validator configured, the finished design is converted to DEF/LEF (every macro
         FIXED, every cell PLACED) and measured, and `ppa.json` is written to `output_dir`.

    The cell placer comes from `EnvironmentSpec.physical`, never from a default: a PPA record
    naming no cell placer next to cells somebody placed would not be attributable.
    """
    from placax.extras.mst import hpwl_wirelength
    from placax.netlist import detect_format
    from placax.netlist.bookshelf import parse_all_node_sizes, parse_nets, parse_pl_positions
    from placax.netlist.def_export import export_bookshelf_as_def

    config = built.config
    physical = config.environment.physical
    benchmark_dir = config.environment.benchmark.path.resolve()
    design_format = detect_format(benchmark_dir)
    if design_format not in (NetlistFormat.BOOKSHELF, NetlistFormat.DEF):
        raise NotImplementedError(
            f"{benchmark_dir} is {design_format.value}; the full flow needs a design with its "
            f"standard cells - Bookshelf or DEF/LEF. A protobuf netlist is clustered and has none."
        )
    if physical.cell_placer is None:
        raise ValueError(
            f"experiment {config.name!r} names no cell placer, so its standard cells have nowhere "
            f"to come from. Add one to EnvironmentSpec.physical, e.g. Spec('dreamplace')."
        )
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # A tool already set on `built` is used as-is, as in evaluate_physical.
    cell_placer = built.cell_placer or build_physical(config, machine)[0]

    if design_format is NetlistFormat.DEF:
        return _place_and_measure_def(
            built, positions, output_dir, machine, orientations, cell_placer, benchmark_dir
        )

    exported = write_placement(built, positions, output_dir, orientations)
    result_pl = cell_placer.place_bookshelf(exported.path, output_dir)

    design = exported.path.stem
    sizes = parse_all_node_sizes(benchmark_dir / f"{design}.nodes")
    placed = parse_pl_positions(result_pl)
    placement = {name: (x, y) for name, (x, y, _fixed) in placed.items() if name in sizes}
    centers = {
        name: (x + sizes[name][0] / 2.0, y + sizes[name][1] / 2.0)
        for name, (x, y) in placement.items()
    }
    full_hpwl = hpwl_wirelength(centers, parse_nets(benchmark_dir / f"{design}.nets", set(sizes)))

    ppa = None
    if physical.validator is not None:
        validate_dir = output_dir / "validate"
        def_path, lef_path = export_bookshelf_as_def(benchmark_dir, validate_dir, placement)
        ppa = evaluate_physical(
            built, def_path, [lef_path], validate_dir, skip_cell_placement=True, machine=machine,
        )
        # The cells WERE placed - by the configured placer, in Bookshelf mode, before conversion.
        ppa = dataclasses.replace(
            ppa, cell_placer=physical.cell_placer.name, legalizer=exported.legalizer,
            max_displacement=exported.max_displacement, off_rows_after=exported.off_rows_after,
        )
        write_ppa(output_dir, ppa)

    return FullDesign(
        result_pl=result_pl, positions=placement, sizes=sizes,
        macro_names=frozenset(built.benchmark.name_to_idx), full_hpwl=float(full_hpwl), ppa=ppa,
    )


def design_lefs(benchmark_dir: pathlib.Path) -> list[pathlib.Path]:
    """A DEF benchmark's LEFs in the order a tool must read them: technology first.

    The technology LEF defines the layers every other LEF refers to, and OpenROAD refuses a cell
    LEF read before it. It is recognised by its content - LAYER definitions and no MACRO - not by
    its name.
    """
    def is_technology(path: pathlib.Path) -> bool:
        text = path.read_text(errors="ignore")
        return "\nLAYER " in text and "\nMACRO " not in text

    lefs = sorted(benchmark_dir.glob("*.lef"))
    return sorted(lefs, key=lambda path: (not is_technology(path), path.name))


def _place_and_measure_def(built, positions, output_dir, machine, orientations, cell_placer,
                           benchmark_dir) -> FullDesign:
    """The DEF half of `place_cells_and_measure`: the design's own LEFs, its own placer."""
    from placax.extras.mst import hpwl_wirelength
    from placax.netlist.def_reader import load_placed_design
    from placax_tools.pipeline import validate_only

    physical = built.config.environment.physical
    lefs = design_lefs(benchmark_dir)
    exported = write_placement(built, positions, output_dir / "placement", orientations)
    placed_def = cell_placer.place(exported.path, lefs, output_dir / "cells")

    placement, sizes, nets = load_placed_design(placed_def, lefs)
    centers = {
        name: (x + sizes[name][0] / 2.0, y + sizes[name][1] / 2.0)
        for name, (x, y) in placement.items() if name in sizes
    }
    full_hpwl = hpwl_wirelength(centers, [
        [pin for pin in net if pin[0] in centers] for net in nets
    ])

    ppa = None
    if physical.validator is not None:
        validator = built.validator or build_physical(built.config, machine)[1]
        measured = validate_only(placed_def, lefs, output_dir / "validate", validator)
        ppa = _physical_result(built.config, measured, placed_def, physical.cell_placer.name)
        ppa = dataclasses.replace(
            ppa, legalizer=exported.legalizer, max_displacement=exported.max_displacement,
            off_rows_after=exported.off_rows_after,
        )
        write_ppa(output_dir, ppa)

    return FullDesign(
        result_pl=placed_def, positions=placement, sizes=sizes,
        macro_names=frozenset(built.benchmark.name_to_idx), full_hpwl=float(full_hpwl), ppa=ppa,
    )
