"""OpenROAD as the cell placer: pins, global placement and detailed placement around FIXED macros.

The cell placer for DEF designs built in a real PDK. DREAMPlace's Bookshelf mode is the reliable
path for the ISPD benchmarks, which carry no technology; a design synthesized and floorplanned by
OpenROAD-flow-scripts is the opposite case - real LEFs, real sites, real tracks - and the tool that
produced its floorplan is the one that places it most predictably. It follows ORFS's own order:

  1. a global placement ignoring the pins, so the pins can then go where the logic wants them
     (ORFS's `3_1_place_gp_skip_io`, `3_2_place_iop`),
  2. the pins, on the configured layers,
  3. the global placement proper, and detailed placement to make it legal.

Every macro the agent placed arrives FIXED and stays where it is. What changes the result -
target density, pin layers, the placer's seed - is a constructor argument and so part of the
config; where OpenROAD lives is not.
"""
import pathlib

from placax_tools.cell_placer import CellPlacer
from placax_tools.openroad.runner import openroad_command, resolve_path, run_openroad

PLACED_DEF_NAME = "placed.def"


def _quote(path) -> str:
    return "{" + str(path) + "}"


def build_placement_script(
    def_path: pathlib.Path,
    lef_paths: list[pathlib.Path],
    placed_def: pathlib.Path,
    density: float = 0.7,
    pin_hor_layers: str | None = None,
    pin_ver_layers: str | None = None,
    routing_layers: str | None = None,
    seed: int | None = None,
    io_constraints: str | None = None,
    density_lb_addon: float | None = None,
) -> str:
    """The Tcl that places every unfixed instance of `def_path` and writes `placed_def`.

    `io_constraints` is a Tcl file sourced before the pins are placed (ORFS's IO_CONSTRAINTS -
    excluded edges, pin groups). `density_lb_addon` is ORFS's PLACE_DENSITY_LB_ADDON: when set,
    the target density is the placer's own lower bound plus that fraction of the headroom,
    rather than `density`.
    """
    lines = [f"read_lef {_quote(path)}" for path in lef_paths]
    lines.append(f"read_def {_quote(def_path)}")
    if routing_layers:
        # The placer's routability model reads these; ORFS sets them before any placement.
        lines.append(f"set_routing_layers -signal {routing_layers}")
    if density_lb_addon is not None:
        lines += [
            "set placax_lb [gpl::get_global_placement_uniform_density]",
            f"set placax_density [expr {{$placax_lb + (1.0 - $placax_lb) * {density_lb_addon} "
            f"+ 0.01}}]",
        ]
    else:
        lines.append(f"set placax_density {density}")
    lines.append('puts "PLACAX_METRIC placement_density $placax_density"')
    seed_flag = f" -random_seed {seed}" if seed is not None else ""
    base = f"global_placement -density $placax_density{seed_flag}"
    if pin_hor_layers and pin_ver_layers:
        lines.append(f"{base} -skip_io")
        if io_constraints:
            lines.append(f"source {_quote(io_constraints)}")
        lines += [
            f"place_pins -hor_layers {pin_hor_layers} -ver_layers {pin_ver_layers}",
        ]
    lines += [
        base,
        "detailed_placement",
        f"write_def {_quote(placed_def)}",
        'puts "PLACAX_PLACEMENT_DONE"',
    ]
    return "\n".join(lines) + "\n"


class OpenROADCellPlacer(CellPlacer):
    """Places standard cells with OpenROAD's global and detailed placers."""

    def __init__(
        self,
        density: float = 0.7,
        pin_hor_layers: str | None = None,
        pin_ver_layers: str | None = None,
        routing_layers: str | None = None,
        seed: int | None = None,
        io_constraints: str | None = None,
        density_lb_addon: float | None = None,
        openroad_binary: str = "openroad",
        use_docker: bool = False,
        docker_image: str | None = None,
    ):
        self.density = density
        self.pin_hor_layers = pin_hor_layers
        self.pin_ver_layers = pin_ver_layers
        self.routing_layers = routing_layers
        self.seed = seed
        self.io_constraints = io_constraints
        self.density_lb_addon = density_lb_addon
        self.openroad_binary = openroad_binary
        self.use_docker = use_docker
        self.docker_image = docker_image

    def place(
        self, def_path: pathlib.Path, lef_paths: list[pathlib.Path], output_dir: pathlib.Path
    ) -> pathlib.Path:
        output_dir = pathlib.Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        def_path = resolve_path(def_path)
        lef_paths = [resolve_path(path) for path in lef_paths]
        placed_def = output_dir / PLACED_DEF_NAME
        placed_def.unlink(missing_ok=True)

        script_path = output_dir / "place.tcl"
        script_path.write_text(build_placement_script(
            def_path, lef_paths, placed_def, self.density, self.pin_hor_layers,
            self.pin_ver_layers, self.routing_layers, self.seed, self.io_constraints,
            self.density_lb_addon,
        ))
        command = openroad_command(
            script_path, [def_path, *lef_paths, self.io_constraints], self.openroad_binary, self.use_docker,
            self.docker_image,
        )
        run_openroad(command, output_dir / "place.log")
        if not placed_def.exists():
            raise FileNotFoundError(f"OpenROAD finished without writing {placed_def}; "
                                    f"see {output_dir / 'place.log'}")
        return placed_def


__all__ = ["OpenROADCellPlacer", "build_placement_script"]
