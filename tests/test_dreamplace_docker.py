import pathlib

from placax_tools.dreamplace.docker import (
    DREAMPLACE_IMAGE, MISSING_PYTHON_DEPS, build_command, clone_command, has_pydeps, install_pydeps_command,
    is_built, is_built_with_cuda, is_cloned, run_placer_command,
)


def test_clone_command_uses_recursive_for_submodules() -> None:
    argv = clone_command(pathlib.Path("/tmp/DREAMPlace"))
    assert argv == ["git", "clone", "--recursive", "https://github.com/limbo018/DREAMPlace.git", "/tmp/DREAMPlace"]


def test_build_command_mounts_repo_and_sets_install_prefix() -> None:
    argv = build_command(pathlib.Path("/tmp/DREAMPlace"))
    assert argv[:3] == ["docker", "run", "--rm"]
    assert "-v" in argv and "/tmp/DREAMPlace:/DREAMPlace" in argv
    assert argv[argv.index(DREAMPLACE_IMAGE) + 1 :] == ["bash", "-c", argv[-1]]
    assert "CMAKE_INSTALL_PREFIX=/DREAMPlace/install" in argv[-1]
    assert "--gpus" not in argv  # gpu=False by default


def test_build_command_gpu_flag() -> None:
    argv = build_command(pathlib.Path("/tmp/DREAMPlace"), gpu=True)
    assert "--gpus" in argv and "all" in argv
    # CUDA is decided at configure time and cached with FORCE: a GPU build starts clean.
    assert argv[-1].startswith("rm -rf build && ")
    # The image's CUDA 11.0 rejects DREAMPlace's default 8.6 target.
    assert '-DCMAKE_CUDA_ARCHITECTURES="6.0;6.1;7.0;7.5;8.0"' in argv[-1]
    assert "rm -rf" not in build_command(pathlib.Path("/tmp/DREAMPlace"))[-1]


def test_a_cpu_only_build_is_told_apart_from_a_cuda_one(tmp_path) -> None:
    configure = tmp_path / "install" / "dreamplace" / "configure.py"
    configure.parent.mkdir(parents=True)
    assert not is_built_with_cuda(tmp_path)
    configure.write_text('compile_configurations = {\n        "CUDA_FOUND" : "", \n}\n')
    assert not is_built_with_cuda(tmp_path)
    configure.write_text('compile_configurations = {\n        "CUDA_FOUND" : "TRUE", \n}\n')
    assert is_built_with_cuda(tmp_path)


def test_a_gpu_run_rebuilds_a_cpu_only_checkout(tmp_path, monkeypatch) -> None:
    import subprocess

    from placax_tools.dreamplace.cell_placer import DREAMPlaceCellPlacer

    (tmp_path / ".git").mkdir()
    (tmp_path / "install" / "pydeps").mkdir(parents=True)
    (tmp_path / "install" / "dreamplace").mkdir()
    (tmp_path / "install" / "dreamplace" / "Placer.py").write_text("")
    (tmp_path / "install" / "dreamplace" / "configure.py").write_text('"CUDA_FOUND" : "",')
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, check: calls.append(argv))

    DREAMPlaceCellPlacer(tmp_path, gpu=True, use_docker=True)._run_dreamplace(tmp_path / "c.json")
    assert any("cmake" in argv[-1] for argv in calls if argv[:2] == ["docker", "run"])
    calls.clear()
    DREAMPlaceCellPlacer(tmp_path, gpu=False, use_docker=True)._run_dreamplace(tmp_path / "c.json")
    assert not any("cmake" in str(argv[-1]) for argv in calls), "a CPU run never rebuilds"


def test_run_placer_command_runs_the_installed_placer_on_the_given_config() -> None:
    argv = run_placer_command(pathlib.Path("/tmp/DREAMPlace"), pathlib.Path("/tmp/out/config.json"))
    assert argv[-3:] == ["python", "install/dreamplace/Placer.py", "/tmp/out/config.json"]


def test_run_placer_command_sets_pythonpath_to_pydeps_dir() -> None:
    argv = run_placer_command(pathlib.Path("/tmp/DREAMPlace"), pathlib.Path("/tmp/out/config.json"))
    assert "-e" in argv and "PYTHONPATH=/DREAMPlace/install/pydeps" in argv


def test_install_pydeps_command_targets_the_persisted_dir() -> None:
    argv = install_pydeps_command(pathlib.Path("/tmp/DREAMPlace"))
    script = argv[-1]
    assert "pip install --no-deps --target=/DREAMPlace/install/pydeps" in script
    for dep in MISSING_PYTHON_DEPS:
        assert dep in script


def test_has_pydeps_reflects_disk_state(tmp_path) -> None:
    repo_dir = tmp_path / "DREAMPlace"
    assert not has_pydeps(repo_dir)
    (repo_dir / "install" / "pydeps").mkdir(parents=True)
    assert has_pydeps(repo_dir)


def test_run_placer_command_mounts_extra_dirs_at_identical_host_paths() -> None:
    argv = run_placer_command(
        pathlib.Path("/tmp/DREAMPlace"), pathlib.Path("/tmp/out/config.json"),
        extra_mounts=(pathlib.Path("/home/paulin/placax"),),
    )
    assert "/home/paulin/placax:/home/paulin/placax" in argv


def test_is_cloned_and_is_built_reflect_disk_state(tmp_path) -> None:
    repo_dir = tmp_path / "DREAMPlace"
    assert not is_cloned(repo_dir)
    assert not is_built(repo_dir)

    (repo_dir / ".git").mkdir(parents=True)
    assert is_cloned(repo_dir)
    assert not is_built(repo_dir)

    (repo_dir / "install" / "dreamplace").mkdir(parents=True)
    (repo_dir / "install" / "dreamplace" / "Placer.py").write_text("")
    assert is_built(repo_dir)
