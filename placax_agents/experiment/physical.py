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

**Still not verified end to end.** No DEF/LEF design ships with this repo and OpenROAD is not a
dependency, so the real binaries have never been driven through this path - the composition, the
export, the TCL generation and the output parsing all have tests, but the numbers themselves are
unproven. The Bookshelf benchmarks under `benchmarks/` still cannot reach the validator, since
OpenROAD reads no Bookshelf; their placements export to `.pl`/`.aux` for DREAMPlace instead.
That is stated here rather than left for a reader to discover.
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

    return PhysicalResult(
        full_hash=config.full_hash(),
        cell_placer=None if skip_cell_placement else physical.cell_placer.name,
        validator=physical.validator.name,
        def_path=str(placed_def),
        design_area=ppa.design_area,
        utilization_pct=ppa.utilization_pct,
        timing_slack=ppa.timing_slack,
        routed_wirelength=ppa.routed_wirelength,
        via_count=ppa.via_count,
        drc_violations=ppa.drc_violations,
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
