"""Shared locations of the real benchmarks that ship with the repo, plus cached loaders.

Every real-scale test used to carry its own hardcoded absolute path to a benchmark directory
outside the repo, so all of them silently skipped on any machine but the one they were written
on. They point here instead now: the benchmarks are committed under benchmarks/, so these
tests run wherever the repo is checked out.

The loaders are lru_cached because adaptec1's .nets file is 35 MB - parsing it once per test
module would dominate the suite's runtime, and the parse is pure, so one result is safe to share.
"""
import functools
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ADAPTEC1 = REPO_ROOT / "benchmarks" / "adaptec1"
BIGBLUE1 = REPO_ROOT / "benchmarks" / "bigblue1"
ARIANE133_PROTOBUF = REPO_ROOT / "benchmarks" / "ariane133" / "netlist.pb.txt"


@functools.lru_cache(maxsize=None)
def load_real_netlist(benchmark_dir: pathlib.Path):
    """Cached load_netlist() - returns the same (macro_sizes, nets) objects to every caller.

    Treat the result as read-only: it is shared across every test in the session.
    """
    from placax.netlist import load_netlist

    return load_netlist(benchmark_dir)


@functools.lru_cache(maxsize=None)
def load_real_padded(benchmark_dir: pathlib.Path):
    """Cached (name_to_idx, sizes_array, padded_pin_idx, padded_pin_offset, valid_mask)."""
    from placax.netlist.padding import build_padded_arrays

    macro_sizes, nets = load_real_netlist(benchmark_dir)
    return build_padded_arrays(macro_sizes, nets)
