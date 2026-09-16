"""Runs one OpenROAD Tcl script - on this host, or in the pinned ORFS image - and keeps its log.

Shared by every OpenROAD-backed tool here (the validator, the cell placer), so there is one place
that knows how the binary is invoked, which directories a container needs, and what a failure
looks like to the caller.
"""
import pathlib
import subprocess


def resolve_path(path) -> pathlib.Path:
    """A host path made absolute; an in-image path (one that does not exist here) left alone."""
    path = pathlib.Path(path)
    return path.resolve() if path.exists() else path


def openroad_command(
    script_path: pathlib.Path,
    paths,
    openroad_binary: str = "openroad",
    use_docker: bool = False,
    docker_image: str | None = None,
) -> list[str]:
    """The argv that runs `script_path`. `paths` are every file the script reads or writes."""
    if not use_docker:
        return [openroad_binary, "-no_splash", "-exit", str(script_path)]
    from placax_tools.openroad.docker import OPENROAD_IMAGE, host_mounts, run_openroad_command

    mounts = host_mounts([script_path, *paths])
    return run_openroad_command(script_path, mounts, docker_image or OPENROAD_IMAGE)


def run_openroad(command: list[str], log_path: pathlib.Path) -> str:
    """Runs OpenROAD, keeps its full log at `log_path`, returns it.

    An uncaught Tcl error exits 1 - reading the design failed, which nothing downstream can
    recover from - and is raised with the tool's own last lines rather than a bare exit code.
    """
    result = subprocess.run(command, capture_output=True, text=True)
    log = result.stdout + (("\n" + result.stderr) if result.stderr else "")
    log_path.write_text(log)
    if result.returncode != 0:
        tail = "\n".join(log.strip().splitlines()[-15:])
        raise subprocess.CalledProcessError(
            result.returncode, command, output=log,
            stderr=f"OpenROAD failed; last lines of {log_path}:\n{tail}",
        )
    return log


__all__ = ["openroad_command", "resolve_path", "run_openroad"]
