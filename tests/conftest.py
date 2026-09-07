"""Puts the repo root on sys.path so tests can import `tests.real_benchmarks` (and the packages
under test) no matter which directory pytest is invoked from."""
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
