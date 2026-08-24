"""DREAMPlace-specific CellPlacer implementation. Matches the
architecture: placax's kernel only does macro placement; cell
placement is a genuinely different problem (analytical optimization
over potentially millions of cells, not RL) handed to an external,
swappable tool.

DREAMPlace's real config format confirmed directly from its own GitHub
issues (actual parsed configs, not guessed): def_input/lef_input are
real fields, invoked via `python dreamplace/Placer.py config.json`.
Not verified end-to-end in this environment - DREAMPlace isn't
installed here (a real, complex CUDA-optional build) - the config
generation and subprocess invocation are built to the documented
format, but only actually running DREAMPlace confirms they work."""
import json
import pathlib
import subprocess

from placax_tools.cell_placer import CellPlacer


def build_dreamplace_config(
    def_path: pathlib.Path,
    lef_paths: list[pathlib.Path],
    output_dir: pathlib.Path,
    gpu: bool = False,
    target_density: float = 1.0,
) -> dict:
    """Real DREAMPlace config fields, confirmed against actual parsed
    configs from DREAMPlace's own GitHub issues - not guessed. Kept as
    a standalone, independently testable function - DREAMPlaceCellPlacer
    just calls it."""
    return {
        "def_input": str(def_path),
        "lef_input": ";".join(str(p) for p in lef_paths),
        "verilog_input": "",
        "gpu": 1 if gpu else 0,
        "num_bins_x": 512,
        "num_bins_y": 512,
        "global_place_stages": [
            {
                "num_bins_x": 512,
                "num_bins_y": 512,
                "iteration": 1000,
                "learning_rate": 0.01,
                "wirelength": "weighted_average",
                "optimizer": "nesterov",
            }
        ],
        "target_density": target_density,
        "density_weight": 8e-5,
        "random_seed": 1000,
        "result_dir": str(output_dir),
        "scale_factor": 1.0,
        "ignore_net_degree": 100,
        "enable_fillers": 1,
        "global_place_flag": 1,
        "legalize_flag": 1,
        "detailed_place_flag": 1,
        "stop_overflow": 0.07,
        "dtype": "float32",
        "plot_flag": 0,
    }


class DREAMPlaceCellPlacer(CellPlacer):
    """Default CellPlacer implementation. Requires a real DREAMPlace
    install - dreamplace_root/gpu/target_density are DREAMPlace-specific
    and live here in __init__, not in place()'s signature, so any other
    CellPlacer subclass can have completely different construction
    needs while still being called identically via place().

    extra_config overrides/adds any other DREAMPlace config field (e.g.
    num_bins_x, random_seed, density_weight) on top of the defaults in
    build_dreamplace_config - more scalable than enumerating every real
    DREAMPlace knob as its own constructor parameter, and stays correct
    even for config fields added in a future DREAMPlace version this
    code doesn't know about yet. python_executable defaults to "python",
    but some installs need "python3" or a specific venv's interpreter."""

    def __init__(
        self,
        dreamplace_root: pathlib.Path,
        gpu: bool = False,
        target_density: float = 1.0,
        extra_config: dict | None = None,
        python_executable: str = "python",
    ):
        self.dreamplace_root = dreamplace_root
        self.gpu = gpu
        self.target_density = target_density
        self.extra_config = extra_config or {}
        self.python_executable = python_executable

    def place(
        self, def_path: pathlib.Path, lef_paths: list[pathlib.Path], output_dir: pathlib.Path
    ) -> pathlib.Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        config = build_dreamplace_config(
            def_path, lef_paths, output_dir, gpu=self.gpu, target_density=self.target_density
        )
        config.update(self.extra_config)
        config_path = output_dir / "dreamplace_config.json"
        config_path.write_text(json.dumps(config, indent=2))

        subprocess.run(
            [self.python_executable, "dreamplace/Placer.py", str(config_path)],
            cwd=self.dreamplace_root,
            check=True,
        )

        result_def = output_dir / f"{def_path.stem}.gp.def"
        if not result_def.exists():
            raise FileNotFoundError(
                f"DREAMPlace ran but expected output {result_def} wasn't produced - "
                "check result_dir naming matches your DREAMPlace version"
            )
        return result_def
