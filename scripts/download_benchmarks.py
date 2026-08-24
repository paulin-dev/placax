"""Fetches benchmarks into benchmarks/<name>/. Meant to run once, by hand -
benchmarks likely get bundled into the repo directly rather than re-run
in CI."""
import enum
import gzip
import pathlib
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass


class Format(enum.Enum):
    """What a downloaded benchmark actually is - these are not interchangeable."""

    ISPD05_NETLIST = "ready-to-place Bookshelf netlist, no synthesis needed"
    PROTOBUF = "ready-to-place Circuit Training netlist, real pin geometry"
    UNSYNTHESIZED_VERILOG = "behavioral RTL - needs synth + format conversion, not built yet"


@dataclass(frozen=True, slots=True)
class Benchmark:
    url: str
    format: Format


# URLs as published on ispd.cc - the /benchmarks/ subpath is genuinely
# inconsistent between designs on the original site, not a typo here.
BENCHMARKS: dict[str, Benchmark] = {
    "adaptec1": Benchmark(
        "http://www.ispd.cc/contests/05/ispd05-contest/adaptec1.tar.gz", Format.ISPD05_NETLIST
    ),
    "adaptec2": Benchmark(
        "http://www.ispd.cc/contests/05/ispd05-contest/benchmarks/adaptec2.tar.gz",
        Format.ISPD05_NETLIST,
    ),
    "adaptec3": Benchmark(
        "http://www.ispd.cc/contests/05/ispd05-contest/adaptec3.tar.gz", Format.ISPD05_NETLIST
    ),
    "adaptec4": Benchmark(
        "http://www.ispd.cc/contests/05/ispd05-contest/benchmarks/adaptec4.tar.gz",
        Format.ISPD05_NETLIST,
    ),
    "bigblue1": Benchmark(
        "http://www.ispd.cc/contests/05/ispd05-contest/benchmarks/bigblue1.tar.gz",
        Format.ISPD05_NETLIST,
    ),
    "bigblue2": Benchmark(
        "http://www.ispd.cc/contests/05/ispd05-contest/benchmarks/bigblue2.tar.gz",
        Format.ISPD05_NETLIST,
    ),
    "bigblue3": Benchmark(
        "http://www.ispd.cc/contests/05/ispd05-contest/benchmarks/bigblue3.tar.gz",
        Format.ISPD05_NETLIST,
    ),
    "bigblue4": Benchmark(
        "http://www.ispd.cc/contests/05/ispd05-contest/benchmarks/bigblue4.tar.gz",
        Format.ISPD05_NETLIST,
    ),
    # ariane133, from AlphaChip/Circuit Training's own repo, not TILOS's
    # DEF/LEF (which has zero real pin geometry - every RECT is a
    # degenerate point, likely anonymized PDK data) or ORFS's RTL (needs
    # synthesis we can't complete in reasonable time). This source has
    # 100% real, nonzero pin offsets on every macro-to-macro net, and its
    # own macro count (133) independently matches the published number.
    "ariane133": Benchmark(
        "https://raw.githubusercontent.com/google-research/circuit_training/"
        "main/circuit_training/environment/test_data/ariane/netlist.pb.txt",
        Format.PROTOBUF,
    ),
}

BENCHMARKS_DIR = pathlib.Path("benchmarks")


def _flatten_nested_dir(out_dir: pathlib.Path, name: str) -> None:
    """Some ISPD05 tarballs nest an extra <name>/ dir inside; others don't."""
    nested = out_dir / name
    if nested.is_dir():
        for item in nested.iterdir():
            item.rename(out_dir / item.name)
        nested.rmdir()


def _degzip_all(out_dir: pathlib.Path) -> None:
    """ISPD05 files (aux.gz, nodes.gz, ...) are gzipped a second time inside the tar."""
    for gz_file in out_dir.glob("*.gz"):
        with gzip.open(gz_file, "rb") as f_in, open(gz_file.with_suffix(""), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_file.unlink()


def _fetch_ispd05_netlist(url: str, out_dir: pathlib.Path, name: str) -> None:
    tar_path = out_dir.parent / f"{name}.tar.gz"
    urllib.request.urlretrieve(url, tar_path)
    with tarfile.open(tar_path) as tar:
        tar.extractall(out_dir, filter="data")
    tar_path.unlink()
    _flatten_nested_dir(out_dir, name)
    _degzip_all(out_dir)


def _fetch_single_file(url: str, out_dir: pathlib.Path) -> None:
    urllib.request.urlretrieve(url, out_dir / pathlib.Path(url).name)


def download_benchmark(name: str, dest_dir: pathlib.Path = BENCHMARKS_DIR) -> None:
    """Fetch one benchmark, skipping if already present."""
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}, available: {list(BENCHMARKS)}")

    out_dir = dest_dir / name
    if out_dir.exists() and any(out_dir.iterdir()):
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark = BENCHMARKS[name]
    match benchmark.format:
        case Format.ISPD05_NETLIST:
            _fetch_ispd05_netlist(benchmark.url, out_dir, name)
        case Format.PROTOBUF | Format.UNSYNTHESIZED_VERILOG:
            _fetch_single_file(benchmark.url, out_dir)
        case _:
            raise ValueError(f"no fetcher registered for {benchmark.format}")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("benchmarks", nargs="*", default=list(BENCHMARKS))
    args = parser.parse_args()

    unknown = [name for name in args.benchmarks if name not in BENCHMARKS]
    if unknown:
        print(f"unknown benchmark(s): {unknown}, available: {list(BENCHMARKS)}", file=sys.stderr)
        sys.exit(1)

    for name in args.benchmarks:
        benchmark = BENCHMARKS[name]
        print(f"fetching {name} ({benchmark.format.value})...")
        download_benchmark(name)
        print(f"  -> {BENCHMARKS_DIR / name}")
