"""Fetches benchmarks into benchmarks/<name>/. Meant to run once, by hand."""
import enum
import gzip
import pathlib
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass

from placax.log import Log


class Format(enum.Enum):
    """What a downloaded benchmark actually is - these are not interchangeable."""

    ISPD05_NETLIST = "ready-to-place Bookshelf netlist, no synthesis needed"
    PROTOBUF = "ready-to-place Circuit Training netlist, real pin geometry"
    UNSYNTHESIZED_VERILOG = "behavioral RTL - needs synth + format conversion, not built yet"


@dataclass(frozen=True, slots=True)
class Benchmark:
    url: str
    format: Format


# URLs as published on ispd.cc - the /benchmarks/ subpath genuinely varies between designs.
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
    # ariane133 from Circuit Training's own repo: 100% real, nonzero pin
    # offsets, unlike TILOS's DEF/LEF (degenerate point RECTs).
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
        # Move everything up one level so files always end up directly in out_dir.
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
    """Downloads the tarball, extracts it, flattens and degzips in place."""
    # 1. Download the tarball next to (not inside) the benchmark's own dir.
    tar_path = out_dir.parent / f"{name}.tar.gz"
    urllib.request.urlretrieve(url, tar_path)
    # 2. Extract it, then discard the archive - we only want the contents.
    with tarfile.open(tar_path) as tar:
        tar.extractall(out_dir, filter="data")
    tar_path.unlink()
    # 3. Normalize the directory layout and undo the extra gzip some ISPD05 benchmarks apply.
    _flatten_nested_dir(out_dir, name)
    _degzip_all(out_dir)


def _fetch_single_file(url: str, out_dir: pathlib.Path) -> None:
    """Downloads a single file (e.g. a .pb.txt netlist) with no extraction needed."""
    urllib.request.urlretrieve(url, out_dir / pathlib.Path(url).name)


def download_benchmark(name: str, dest_dir: pathlib.Path = BENCHMARKS_DIR) -> None:
    """Fetches one benchmark, skipping if already present."""
    # 1. Validate the name up front so typos fail fast with a helpful message.
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}, available: {list(BENCHMARKS)}")

    # 2. Skip re-downloading if this benchmark was already fetched before.
    out_dir = dest_dir / name
    if out_dir.exists() and any(out_dir.iterdir()):
        return  # already fetched
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Dispatch to the right fetch strategy for this benchmark's format.
    benchmark = BENCHMARKS[name]
    match benchmark.format:  # each format needs a different fetch strategy
        case Format.ISPD05_NETLIST:
            _fetch_ispd05_netlist(benchmark.url, out_dir, name)
        case Format.PROTOBUF | Format.UNSYNTHESIZED_VERILOG:
            _fetch_single_file(benchmark.url, out_dir)
        case _:
            raise ValueError(f"no fetcher registered for {benchmark.format}")


if __name__ == "__main__":
    import argparse
    import sys

    Log.configure()

    # 1. Parse the requested benchmark names; default to fetching them all.
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmarks", nargs="*", default=list(BENCHMARKS))
    args = parser.parse_args()

    # 2. Fail fast on typos rather than partially downloading then erroring.
    unknown = [name for name in args.benchmarks if name not in BENCHMARKS]
    if unknown:
        Log.error(f"unknown benchmark(s): {unknown}, available: {list(BENCHMARKS)}")
        sys.exit(1)

    # 3. Fetch each requested benchmark, one at a time.
    for name in args.benchmarks:
        benchmark = BENCHMARKS[name]
        Log.info(f"fetching {name} ({benchmark.format.value})...")
        download_benchmark(name)
        Log.info(f"  -> {BENCHMARKS_DIR / name}")
