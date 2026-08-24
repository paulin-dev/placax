"""Guards the placax-before-jax import ordering _device.py depends on -
found broken in 6 real files this session (including a script written
in the same conversation as the fix), not a hypothetical concern.
Manual vigilance already failed once; this checks every file, always."""
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).parent.parent
_JAX_IMPORT_RE = re.compile(r"^(import jax\b|from jax\b|import jax\.numpy\b)", re.MULTILINE)
_PLACAX_IMPORT_RE = re.compile(r"^(import placax(_\w+)?\b|from placax(_\w+)?\b)", re.MULTILINE)

# placax/_device.py and placax/__init__.py are the definitions themselves,
# not consumers - excluded, they can't import "placax" before it exists.
_EXCLUDED = {"_device.py", "__init__.py"}


def _files_importing_both() -> list[pathlib.Path]:
    files = []
    for pattern in (
        "placax/**/*.py", "placax_agents/**/*.py", "placax_tools/**/*.py",
        "scripts/**/*.py", "tests/**/*.py",
    ):
        for path in REPO_ROOT.glob(pattern):
            if path.name in _EXCLUDED:
                continue
            text = path.read_text()
            if _JAX_IMPORT_RE.search(text) and _PLACAX_IMPORT_RE.search(text):
                files.append(path)
    return files


def test_placax_always_imported_before_jax() -> None:
    violations = []
    for path in _files_importing_both():
        text = path.read_text()
        first_jax = _JAX_IMPORT_RE.search(text).start()
        first_placax = _PLACAX_IMPORT_RE.search(text).start()
        if first_jax < first_placax:
            violations.append(str(path.relative_to(REPO_ROOT)))

    assert not violations, (
        f"these files import jax before placax, meaning _device.py's env vars "
        f"never get a chance to apply: {violations}"
    )


def test_this_check_actually_detects_a_real_violation() -> None:
    # Regression test for the checker itself: prove it would have caught
    # the actual bug found this session, not just that it passes today.
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        bad_file = pathlib.Path(tmpdir) / "placax" / "bad.py"
        bad_file.parent.mkdir(parents=True)
        bad_file.write_text("import jax\nfrom placax.core import reset\n")

        text = bad_file.read_text()
        first_jax = _JAX_IMPORT_RE.search(text).start()
        first_placax = _PLACAX_IMPORT_RE.search(text).start()
        assert first_jax < first_placax  # confirms this pattern IS flagged as a violation


def test_placax_import_regex_also_matches_placax_agents() -> None:
    # Regression test for a real, separate bug: \b does not create a
    # word boundary at an underscore, so the regex never matched
    # "placax_agents" imports at all - 4 real files (2 pre-existing,
    # unnoticed the whole time) had this exact ordering bug completely
    # invisible to the checker until this was fixed.
    assert _PLACAX_IMPORT_RE.search("from placax_agents.training.algorithm.normalize import normalize_advantages")
    assert _PLACAX_IMPORT_RE.search("import placax_agents")


def test_package_inits_import_device_first() -> None:
    # A real, distinct gap the per-file checker structurally can't see:
    # a file that imports jax but nothing from placax at all (5 real
    # files found this way in placax_agents) isn't in the "imports both"
    # set above, so it's invisible to that check. The only real
    # safeguard is each package's own __init__.py importing _device
    # first - protecting every file in the package via Python's own
    # parent-initializes-first guarantee, not per-file vigilance.
    for init_path in (
        REPO_ROOT / "placax" / "__init__.py",
        REPO_ROOT / "placax_agents" / "__init__.py",
        REPO_ROOT / "placax_tools" / "__init__.py",
    ):
        text = init_path.read_text()
        assert re.search(r"from placax import _device", text), (
            f"{init_path} must import _device first - otherwise any file in this "
            f"package that imports jax but nothing else from placax (a real, "
            f"confirmed case) can bypass _device's env var setup entirely"
        )
