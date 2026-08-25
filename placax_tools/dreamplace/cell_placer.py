"""DREAMPlace-specific CellPlacer. Not verified end-to-end here -
DREAMPlace isn't installed in this environment."""
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
    """Builds the DREAMPlace config dict (standalone for testability)."""
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
    """Default CellPlacer, requiring a real DREAMPlace install.
    extra_config overrides/adds any DREAMPlace config field."""

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

    def _write_config(
        self, def_path: pathlib.Path, lef_paths: list[pathlib.Path], output_dir: pathlib.Path
    ) -> pathlib.Path:
        """Builds and writes the config DREAMPlace itself reads."""
        config = build_dreamplace_config(
            def_path, lef_paths, output_dir, gpu=self.gpu, target_density=self.target_density
        )
        config.update(self.extra_config)
        config_path = output_dir / "dreamplace_config.json"
        config_path.write_text(json.dumps(config, indent=2))
        return config_path

    def _run_dreamplace(self, config_path: pathlib.Path) -> None:
        """Runs DREAMPlace as an external process."""
        subprocess.run(
            [self.python_executable, "dreamplace/Placer.py", str(config_path)],
            cwd=self.dreamplace_root,
            check=True,
        )

    def _expect_result_def(self, def_path: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
        """Confirms DREAMPlace actually produced the DEF we expect."""
        result_def = output_dir / f"{def_path.stem}.gp.def"
        if not result_def.exists():
            raise FileNotFoundError(
                f"DREAMPlace ran but expected output {result_def} wasn't produced - "
                "check result_dir naming matches your DREAMPlace version"
            )
        return result_def

    def place(
        self, def_path: pathlib.Path, lef_paths: list[pathlib.Path], output_dir: pathlib.Path
    ) -> pathlib.Path:
        """Writes the config, runs DREAMPlace, returns the placed DEF."""
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = self._write_config(def_path, lef_paths, output_dir)
        self._run_dreamplace(config_path)
        return self._expect_result_def(def_path, output_dir)
