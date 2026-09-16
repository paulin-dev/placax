"""Runs OpenROAD through the official OpenROAD-flow-scripts image, the way DREAMPlace runs through its own.

Building OpenROAD from source is a long, dependency-heavy job, and the validator was the one box in
this project's architecture that no machine here had ever driven. The ORFS image ships a prebuilt
`openroad` binary - and, more usefully, the open PDKs' real technology files (Nangate45, sky130hd,
asap7: tech LEFs, cell LEFs, liberty timing libraries), which is what makes a PPA number a number
about a process rather than about a file.

**The image tag is pinned, and that is a reproducibility decision, not caution.** ORFS publishes a
new image several times a week, and OpenROAD's placer, router and timer change between them. Two
labs validating one placement with `:latest` a week apart would be measuring with two different
tools while their `ppa.json` files claimed the same one. The tag is recorded in every result.

**Paths bind at the same place inside the container as on the host**, exactly like
`dreamplace/docker.py`, so the absolute paths already written into a TCL script resolve unchanged
and nothing needs translating. A path that does NOT exist on the host is assumed to live inside
the image - that is how a script names `/OpenROAD-flow-scripts/flow/platforms/sky130hd/...`.
"""
import os
import pathlib
import subprocess

OPENROAD_IMAGE = "openroad/orfs:26Q3-584-gaabf2a397"
"""Pinned ORFS image. Its OpenROAD reports itself as `26Q3-2130-g90e29809c3`."""

OPENROAD_BINARY_IN_IMAGE = "/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad"
"""Not on the image's PATH - its entrypoint expects `source env.sh` first - so it is named directly."""

PLATFORMS_IN_IMAGE = pathlib.PurePosixPath("/OpenROAD-flow-scripts/flow/platforms")
"""The open PDKs the image carries. See `platform_files`."""

PLATFORMS = {
    # platform: (tech LEF, cell LEF, liberty, a routing layer for wire parasitics)
    "sky130hd": ("lef/sky130_fd_sc_hd.tlef", "lef/sky130_fd_sc_hd_merged.lef",
                 "lib/sky130_fd_sc_hd__tt_025C_1v80.lib", "met2"),
    "nangate45": ("lef/NangateOpenCellLibrary.tech.lef", "lef/NangateOpenCellLibrary.macro.mod.lef",
                  "lib/NangateOpenCellLibrary_typical.lib", "metal3"),
}
"""The technologies a real design can be measured against without anything installed on the host.

Only meaningful for a design built IN that technology - its cells have to be the library's cells.
A Bookshelf benchmark converted by `placax/netlist/def_export.py` is not: its cell library is
derived from the netlist and its layer is called `metal1`, so it is measured with its own LEF."""


def platform_files(platform: str) -> dict[str, str]:
    """{tech_lef, cell_lef, liberty, wire_rc_layer} for one of the image's platforms, as in-image paths."""
    if platform not in PLATFORMS:
        raise KeyError(
            f"unknown platform {platform!r}; the pinned image carries {', '.join(sorted(PLATFORMS))}"
        )
    tech, cells, liberty, layer = PLATFORMS[platform]
    root = PLATFORMS_IN_IMAGE / platform
    return {
        "tech_lef": str(root / tech), "cell_lef": str(root / cells),
        "liberty": str(root / liberty), "wire_rc_layer": layer,
    }


def host_mounts(paths) -> list[pathlib.Path]:
    """The host directories a container needs, for whichever of `paths` exist on the host.

    One mount per distinct directory, deduplicated so a design and its LEF side by side cost one
    `-v`. Paths that do not exist here are left alone: they name files inside the image.
    """
    mounts: list[pathlib.Path] = []
    for raw in paths:
        if raw is None:
            continue
        path = pathlib.Path(raw)
        if not path.exists():
            continue
        directory = (path if path.is_dir() else path.parent).resolve()
        if not any(directory == m or directory.is_relative_to(m) for m in mounts):
            mounts = [m for m in mounts if not m.is_relative_to(directory)] + [directory]
    return mounts


def run_openroad_command(
    script_path: pathlib.Path,
    mounts: list[pathlib.Path],
    image: str = OPENROAD_IMAGE,
) -> list[str]:
    """`docker run` argv executing `script_path` with the image's OpenROAD.

    Runs as the calling user, so the reports it writes into bind-mounted directories are owned by
    whoever ran the experiment rather than by root - otherwise every result directory this touches
    would need `sudo` to clean up.
    """
    argv = ["docker", "run", "--rm"]
    if hasattr(os, "getuid"):
        argv += ["--user", f"{os.getuid()}:{os.getgid()}"]
    # OpenROAD writes a history file and Tcl caches under $HOME; a non-root user in this image has
    # no home directory, so point it somewhere that exists and is writable.
    argv += ["-e", "HOME=/tmp"]
    for mount in mounts:
        argv += ["-v", f"{mount}:{mount}"]
    argv += [image, OPENROAD_BINARY_IN_IMAGE, "-no_splash", "-exit", str(script_path)]
    return argv


def image_available(image: str = OPENROAD_IMAGE) -> bool:
    """Whether Docker is reachable and the pinned image is already pulled.

    Checked rather than pulled implicitly: the image is 1.6 GB, and a validation run that quietly
    starts a download is a validation run that looks hung.
    """
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, timeout=30
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def pull_command(image: str = OPENROAD_IMAGE) -> list[str]:
    return ["docker", "pull", image]


__all__ = [
    "OPENROAD_BINARY_IN_IMAGE", "OPENROAD_IMAGE", "PLATFORMS", "host_mounts", "image_available",
    "platform_files", "pull_command", "run_openroad_command",
]
