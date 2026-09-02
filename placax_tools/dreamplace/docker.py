"""Runs DREAMPlace via its official Docker image (limbo018/dreamplace:cuda) instead of requiring a
matching GCC/Boost/Bison/Flex/CMake/PyTorch toolchain on the host - DREAMPlace's C++/CUDA extensions are
compiled from source, and there is no real pip package (the "dreamplace" PyPI entry is an unreleased
placeholder). The image itself is only a build ENVIRONMENT (its own Dockerfile installs the toolchain,
not DREAMPlace's source - confirmed against the real Dockerfile on GitHub), so DREAMPlace is cloned once
to disk and built INSIDE the container on first use; every command that does so lives here, in one file,
not in ad hoc shell history."""
import pathlib

DREAMPLACE_IMAGE = "limbo018/dreamplace:cuda"
DREAMPLACE_REPO_URL = "https://github.com/limbo018/DREAMPlace.git"
INSTALLED_PLACER = "install/dreamplace/Placer.py"  # CMAKE_INSTALL_PREFIX defaults to ./install (in-tree)
PYDEPS_DIR = "install/pydeps"
MISSING_PYTHON_DEPS = ("torch_optimizer==0.3.0", "ncg_optimizer==0.2.2", "pytorch-ranger==0.1.1")
"""torch_optimizer/ncg_optimizer are in DREAMPlace's own requirements.txt but NOT installed by the image's
Dockerfile (confirmed against both files directly) - needed at runtime by NonLinearPlace.py, not just at
build time. pytorch-ranger is torch_optimizer's own (undeclared-in-requirements.txt) dependency - installed
explicitly here since install_pydeps_command uses --no-deps (see its docstring for why)."""


def is_cloned(repo_dir: pathlib.Path) -> bool:
    return (repo_dir / ".git").exists()


def is_built(repo_dir: pathlib.Path) -> bool:
    return (repo_dir / INSTALLED_PLACER).exists()


def has_pydeps(repo_dir: pathlib.Path) -> bool:
    return (repo_dir / PYDEPS_DIR).exists()


def clone_command(repo_dir: pathlib.Path) -> list[str]:
    """git clone --recursive - DREAMPlace vendors Limbo/Flute/CUB/munkres-cpp/OpenTimer as submodules."""
    return ["git", "clone", "--recursive", DREAMPLACE_REPO_URL, str(repo_dir)]


def _docker_run_argv(
    repo_dir: pathlib.Path, gpu: bool, command: list[str], extra_mounts: tuple[pathlib.Path, ...] = ()
) -> list[str]:
    """Every extra mount binds at the SAME path inside the container as on the host, so absolute host
    paths already written into our config/.aux files (result_dir, aux_input, nodes/nets/wts/scl/pl)
    resolve unchanged - no path-translation logic needed anywhere else in the pipeline."""
    argv = ["docker", "run", "--rm"]
    if gpu:
        argv += ["--gpus", "all"]
    # Every container is ephemeral (--rm), so anything pip-installed only inside it vanishes with it;
    # pointing PYTHONPATH at the bind-mounted PYDEPS_DIR (installed there once, via install_pydeps_command)
    # makes those packages visible on every later run too, the same persistence trick build_command uses
    # for the compiled C++ extensions.
    argv += ["-e", f"PYTHONPATH=/DREAMPlace/{PYDEPS_DIR}"]
    argv += ["-v", f"{repo_dir}:/DREAMPlace", "-w", "/DREAMPlace"]
    for mount in extra_mounts:
        argv += ["-v", f"{mount}:{mount}"]
    argv += [DREAMPLACE_IMAGE] + command
    return argv


def build_command(repo_dir: pathlib.Path, gpu: bool = False) -> list[str]:
    """Builds DREAMPlace INSIDE the container (using ITS gcc/boost/bison/flex/cmake/torch, sidestepping
    whatever the host does or doesn't have), writing compiled output to <repo_dir>/install on the host via
    the bind mount - a one-time cost; every later run_placer_command call reuses it."""
    build_script = (
        "mkdir -p build && cd build && "
        "cmake .. -DCMAKE_INSTALL_PREFIX=/DREAMPlace/install -DPython_EXECUTABLE=$(which python) && "
        "make -j$(nproc) && make install"
    )
    return _docker_run_argv(repo_dir, gpu, ["bash", "-c", build_script])


def install_pydeps_command(repo_dir: pathlib.Path, gpu: bool = False) -> list[str]:
    """Installs MISSING_PYTHON_DEPS into PYDEPS_DIR (inside the bind-mounted repo_dir, so it persists on
    the host disk across --rm containers) - cheap and separate from build_command's expensive C++/CUDA
    compile, so a Python-only dependency fix never forces a full rebuild. --no-deps is essential: without
    it, pip resolves torch_optimizer's own "torch>=1.5.0" dependency and installs a brand-new torch into
    PYTHONPATH, shadowing the image's torch 1.7.1 that the compiled C++ extensions are actually linked
    against - breaking them via an ABI mismatch instead of just adding two small pure-Python packages."""
    pip_script = f"pip install --no-deps --target=/DREAMPlace/{PYDEPS_DIR} " + " ".join(MISSING_PYTHON_DEPS)
    return _docker_run_argv(repo_dir, gpu, ["bash", "-c", pip_script])


def run_placer_command(
    repo_dir: pathlib.Path,
    config_path: pathlib.Path,
    gpu: bool = False,
    extra_mounts: tuple[pathlib.Path, ...] = (),
) -> list[str]:
    """Runs an already-built DREAMPlace's Placer.py on config_path inside the container."""
    return _docker_run_argv(
        repo_dir, gpu, ["python", INSTALLED_PLACER, str(config_path)], extra_mounts=extra_mounts,
    )
