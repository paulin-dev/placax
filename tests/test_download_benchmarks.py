import gzip
import pathlib
import urllib.request

import pytest

from scripts.download_benchmarks import (
    BENCHMARKS,
    _degzip_all,
    _flatten_nested_dir,
    download_benchmark,
)


def test_flatten_nested_dir_handles_nested_structure(tmp_path: pathlib.Path) -> None:
    # Some ISPD05 tarballs nest an extra <name>/ dir inside.
    nested = tmp_path / "adaptec2" / "adaptec2"
    nested.mkdir(parents=True)
    (nested / "adaptec2.aux.gz").write_bytes(b"placeholder")

    out_dir = tmp_path / "adaptec2"
    _flatten_nested_dir(out_dir, "adaptec2")

    assert (out_dir / "adaptec2.aux.gz").exists()
    assert not (out_dir / "adaptec2").exists()


def test_flatten_nested_dir_leaves_flat_structure_untouched(tmp_path: pathlib.Path) -> None:
    # Other ISPD05 tarballs extract flat, no nested subdirectory - no-op expected.
    out_dir = tmp_path / "adaptec1"
    out_dir.mkdir()
    (out_dir / "adaptec1.aux.gz").write_bytes(b"placeholder")

    _flatten_nested_dir(out_dir, "adaptec1")

    assert (out_dir / "adaptec1.aux.gz").exists()


def test_degzip_all_decompresses_and_removes_gz(tmp_path: pathlib.Path) -> None:
    with gzip.open(tmp_path / "adaptec1.aux.gz", "wb") as f:
        f.write(b"aux content")

    _degzip_all(tmp_path)

    assert (tmp_path / "adaptec1.aux").read_bytes() == b"aux content"
    assert not (tmp_path / "adaptec1.aux.gz").exists()


def test_flatten_then_degzip_compose_correctly(tmp_path: pathlib.Path) -> None:
    # The real call order in _fetch_ispd05_netlist: flatten first, then degzip.
    nested = tmp_path / "adaptec2" / "adaptec2"
    nested.mkdir(parents=True)
    with gzip.open(nested / "adaptec2.aux.gz", "wb") as f:
        f.write(b"aux content")

    out_dir = tmp_path / "adaptec2"
    _flatten_nested_dir(out_dir, "adaptec2")
    _degzip_all(out_dir)

    assert (out_dir / "adaptec2.aux").read_bytes() == b"aux content"
    assert not (out_dir / "adaptec2").exists()


def _reachable(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=5)
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _reachable(BENCHMARKS["ariane133"].url), reason="AlphaChip ariane133 source not reachable"
)
def test_download_ariane133(tmp_path: pathlib.Path) -> None:
    download_benchmark("ariane133", tmp_path)
    out_file = tmp_path / "ariane133" / "netlist.pb.txt"
    assert out_file.exists()
    assert out_file.stat().st_size > 1_000_000  # real file is ~14MB


@pytest.mark.skipif(
    not _reachable(BENCHMARKS["adaptec1"].url), reason="ispd.cc not reachable from this network"
)
def test_download_adaptec1(tmp_path: pathlib.Path) -> None:
    download_benchmark("adaptec1", tmp_path)
    extracted = list((tmp_path / "adaptec1").iterdir())
    assert extracted, "adaptec1 tarball extracted nothing"


def test_unknown_benchmark_raises(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="unknown benchmark"):
        download_benchmark("not_a_real_benchmark", tmp_path)


def test_cli_rejects_unknown_names_before_downloading(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/download_benchmarks.py", "adaptec1", "ariane"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "unknown benchmark" in result.stderr
    assert "ariane133" in result.stderr  # the actual correct name should be suggested


def test_skips_already_downloaded(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "adaptec1"
    out_dir.mkdir()
    (out_dir / "already_here.txt").write_text("x")

    def _fail_if_called(*a, **kw):
        raise AssertionError("should not attempt a download when files already exist")

    monkeypatch.setattr(urllib.request, "urlretrieve", _fail_if_called)
    download_benchmark("adaptec1", tmp_path)  # should return early, not raise
