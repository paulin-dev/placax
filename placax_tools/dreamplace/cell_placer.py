"""DREAMPlace-specific CellPlacer."""
import json
import pathlib
import subprocess

from placax_tools.cell_placer import CellPlacer
from placax_tools.dreamplace import docker as dreamplace_docker


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


def build_dreamplace_config_bookshelf(
    aux_path: pathlib.Path,
    output_dir: pathlib.Path,
    gpu: bool = False,
    target_density: float = 1.0,
) -> dict:
    """Builds the DREAMPlace config dict for Bookshelf (.aux) input - DREAMPlace's native, best-tested
    format, matched field-for-field against its own real test/ispd2005/adaptec1.json config."""
    return {
        "aux_input": str(aux_path),
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
                "Llambda_density_weight_iteration": 1,
                "Lsub_iteration": 1,
            }
        ],
        "target_density": target_density,
        "density_weight": 8e-5,
        "gamma": 4.0,
        "random_seed": 1000,
        "result_dir": str(output_dir),
        "scale_factor": 1.0,
        "ignore_net_degree": 100,
        "enable_fillers": 1,
        "gp_noise_ratio": 0.025,
        "global_place_flag": 1,
        "legalize_flag": 1,
        "detailed_place_flag": 1,
        "detailed_place_engine": "",
        "detailed_place_command": "",
        "stop_overflow": 0.07,
        "dtype": "float32",
        "plot_flag": 0,
        "random_center_init_flag": 1,
        "gift_init_flag": 0,
        "sort_nets_by_degree": 0,
        "num_threads": 8,
        "deterministic_flag": 1,
    }


class DREAMPlaceCellPlacer(CellPlacer):
    """Default CellPlacer, requiring a real DREAMPlace install; extra_config overrides/adds any config field."""

    def __init__(
        self,
        dreamplace_root: pathlib.Path,
        gpu: bool = False,
        target_density: float = 1.0,
        extra_config: dict | None = None,
        python_executable: str = "python",
        use_docker: bool = False,
        extra_mounts: tuple[pathlib.Path, ...] = (),
    ):
        self.dreamplace_root = dreamplace_root
        self.gpu = gpu
        self.target_density = target_density
        self.extra_config = extra_config or {}
        self.python_executable = python_executable
        # Docker mode: dreamplace_root is where DREAMPlace's source gets cloned (auto-cloned/built on
        # first use, via placax_tools/dreamplace/docker.py, if it isn't already there) instead of a
        # pre-existing local checkout - avoids needing DREAMPlace's GCC/Boost/Bison/Flex/CMake/PyTorch
        # toolchain on the host at all. extra_mounts are additional host directories (e.g. the benchmark
        # dir and the pipeline's output dir) made visible inside the container at their own host path.
        self.use_docker = use_docker
        self.extra_mounts = extra_mounts

    def _write_config(
        self, def_path: pathlib.Path, lef_paths: list[pathlib.Path], output_dir: pathlib.Path
    ) -> pathlib.Path:
        """Builds and writes the config DREAMPlace itself reads."""
        # 1. Start from the standard config for this placement.
        config = build_dreamplace_config(
            def_path, lef_paths, output_dir, gpu=self.gpu, target_density=self.target_density
        )
        # 2. Let any user-supplied overrides win (e.g. tuning knobs per design).
        config.update(self.extra_config)
        # 3. Write it out as JSON - this is the file DREAMPlace's CLI expects.
        config_path = output_dir / "dreamplace_config.json"
        config_path.write_text(json.dumps(config, indent=2))
        return config_path

    def _run_dreamplace(self, config_path: pathlib.Path) -> None:
        """Runs DREAMPlace as an external process - either a local checkout, or the official Docker
        image (limbo018/dreamplace:cuda), cloning/building DREAMPlace inside the container once, on
        first use, if dreamplace_root isn't already a built checkout."""
        if self.use_docker:
            if not dreamplace_docker.is_cloned(self.dreamplace_root):
                self.dreamplace_root.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(dreamplace_docker.clone_command(self.dreamplace_root), check=True)
            if not dreamplace_docker.is_built(self.dreamplace_root):
                subprocess.run(dreamplace_docker.build_command(self.dreamplace_root, gpu=self.gpu), check=True)
            if not dreamplace_docker.has_pydeps(self.dreamplace_root):
                subprocess.run(
                    dreamplace_docker.install_pydeps_command(self.dreamplace_root, gpu=self.gpu), check=True
                )
            subprocess.run(
                dreamplace_docker.run_placer_command(
                    self.dreamplace_root, config_path, gpu=self.gpu, extra_mounts=self.extra_mounts
                ),
                check=True,
            )
            return
        subprocess.run(
            [self.python_executable, "dreamplace/Placer.py", str(config_path)],
            cwd=self.dreamplace_root,
            check=True,
        )

    def _expect_result_def(self, def_path: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
        """Confirms DREAMPlace actually produced the DEF we expect."""
        # DREAMPlace writes into a <result_dir>/<design_name>/ subdirectory (confirmed end-to-end against
        # a real run, not just the README), named after the design with a ".gp.def" suffix.
        result_def = output_dir / def_path.stem / f"{def_path.stem}.gp.def"
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
        # 1. Make sure the output directory exists before anything writes to it.
        output_dir.mkdir(parents=True, exist_ok=True)
        # 2. Prepare the config file DREAMPlace will read.
        config_path = self._write_config(def_path, lef_paths, output_dir)
        # 3. Hand off to the external DREAMPlace process to do the actual placement.
        self._run_dreamplace(config_path)
        # 4. Confirm it produced output and return its path.
        return self._expect_result_def(def_path, output_dir)

    def _write_config_bookshelf(self, aux_path: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
        """Builds and writes the Bookshelf-mode config DREAMPlace itself reads."""
        config = build_dreamplace_config_bookshelf(
            aux_path, output_dir, gpu=self.gpu, target_density=self.target_density
        )
        config.update(self.extra_config)
        config_path = output_dir / "dreamplace_config_bookshelf.json"
        config_path.write_text(json.dumps(config, indent=2))
        return config_path

    def _expect_result_pl(self, aux_path: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
        """Confirms DREAMPlace actually produced the Bookshelf .pl we expect."""
        # DREAMPlace writes into a <result_dir>/<design_name>/ subdirectory (confirmed end-to-end against
        # a real run, not just the README), named after the design (the .aux stem) with a ".gp.pl" suffix.
        result_pl = output_dir / aux_path.stem / f"{aux_path.stem}.gp.pl"
        if not result_pl.exists():
            raise FileNotFoundError(
                f"DREAMPlace ran but expected output {result_pl} wasn't produced - "
                "check result_dir naming matches your DREAMPlace version"
            )
        return result_pl

    def place_bookshelf(self, aux_path: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
        """Bookshelf-native placement: writes the config, runs DREAMPlace, returns the placed .pl - DREAMPlace's
        best-tested input path, and the one our Bookshelf-only benchmarks (no real LEF/DEF) actually need. Not
        part of the generic CellPlacer ABC (which stays DEF/LEF, since that's the only format OpenROAD reads)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = self._write_config_bookshelf(aux_path, output_dir)
        self._run_dreamplace(config_path)
        return self._expect_result_pl(aux_path, output_dir)
