"""The design document has to describe the tree that exists, and a test has to be what says so.

Section 4.2 of the spec presents its file listing as "verified against the repo, not
aspirational". It has drifted from the repository twice anyway - first before v4, which is why
v4's changelog says the section was rewritten to match the shipped code, and then again
afterwards, once `placax_agents/experiment/`, `placax_agents/agents/`, `placax_viz/` and the tool
wrappers appeared without it. Correcting it a third time by hand would only reset the clock on
the same failure, so it is checked here in both directions:

  * every path the docs NAME must exist, and
  * every module the tree SHIPS must be named.

The second direction is the one that actually failed, and the one no amount of careful reading
catches.
"""
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = REPO_ROOT / "docs" / "JAX_Placement_Environment_Spec.md"

DOCS = sorted((REPO_ROOT / "docs").glob("*.md")) + [REPO_ROOT / "README.md"]

PACKAGES = ("placax", "placax_agents", "placax_tools", "placax_viz", "scripts")
"""Checked in both directions against the spec's file listing. `scripts` is in the list
because leaving it out is how the listing came to describe Tier 3 with an ellipsis while
four scripts went unnamed - the same silent drift this test exists to catch one tier up."""

_PATH_RE = re.compile(
    r"(?<![\w./-])((?:placax|placax_agents|placax_tools|placax_viz|scripts|tests)(?:/[\w.\-]+)+)"
)

NOT_YET_BUILT = {
    # Named in the docs precisely as things that do NOT exist yet. Section 5.6 introduces this
    # one as "new file"; keeping it allowlisted rather than deleting the discussion is the point
    # - the spec is allowed to describe future work, as long as it says so.
    "placax_agents/offline.py",
}

PROSE_FALSE_POSITIVES = {
    # Not paths: "placax/jax" is prose for "placax or jax", "tests/CI" for "tests or CI".
    "placax/jax",
    "tests/CI",
}

# Modules that exist but deliberately aren't in the spec's tier listing: private helpers and
# package boilerplate, which would bury the structure the listing exists to show.
UNLISTED_MODULES = {"__init__.py"}


def _named_paths(text: str) -> set[str]:
    return set(_PATH_RE.findall(text)) - NOT_YET_BUILT - PROSE_FALSE_POSITIVES


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_every_repository_path_named_in_the_docs_exists(doc: pathlib.Path) -> None:
    # A doc that points at a file which was renamed or deleted sends a reader looking for
    # something that isn't there, which is worse than not mentioning it.
    missing = sorted(path for path in _named_paths(doc.read_text())
                     if not (REPO_ROOT / path).exists())
    assert not missing, (
        f"{doc.relative_to(REPO_ROOT)} names paths that do not exist: {missing}. Either fix the "
        f"reference, or add it to NOT_YET_BUILT here if the doc is deliberately describing "
        f"something unbuilt."
    )


def _spec_file_tree() -> str:
    """Section 4.2's fenced code block - the 'concrete file structure, as shipped' listing."""
    text = SPEC.read_text()
    start = text.index("placax/                          # Tier 1")
    return text[start:text.index("```", start)]


def _shipped_modules() -> list[pathlib.Path]:
    """Every module in the four packages, minus vendored third-party source and boilerplate."""
    modules = []
    for package in PACKAGES:
        for path in sorted((REPO_ROOT / package).rglob("*.py")):
            relative = path.relative_to(REPO_ROOT)
            # DREAMPlace is an external checkout cloned in by --use_docker, never ours to list.
            if "DREAMPlace" in relative.parts or "__pycache__" in relative.parts:
                continue
            if relative.name in UNLISTED_MODULES:
                continue
            modules.append(relative)
    return modules


def test_the_specs_file_tree_lists_every_shipped_module() -> None:
    # The direction that actually drifted: files appear, and a hand-maintained listing silently
    # stops describing the project while still claiming to.
    tree = _spec_file_tree()
    missing = sorted(
        str(module) for module in _shipped_modules()
        # A module counts as listed if its filename appears anywhere in the block - the listing
        # is indented and grouped by directory, so requiring full paths would be noise.
        if module.name not in tree
    )
    assert not missing, (
        f"docs/JAX_Placement_Environment_Spec.md §4.2 claims to list the shipped tree but "
        f"omits: {missing}. Add them to that listing, or to UNLISTED_MODULES here if they are "
        f"deliberately below the level the section describes."
    )


def test_the_specs_file_tree_names_only_real_directories() -> None:
    # The other direction, scoped to the tree block: a listing entry pointing at a package that
    # no longer exists is the same drift running backwards.
    tree = _spec_file_tree()
    # Only unindented entries: those are repo-root packages. Indented ones are subdirectories
    # named relative to the package above them, which don't exist at the root.
    listed_dirs = set(re.findall(r"^([a-z_]+)/\s", tree, flags=re.MULTILINE))
    missing = sorted(name for name in listed_dirs if not (REPO_ROOT / name).exists())
    assert not missing, f"§4.2 lists directories that do not exist: {missing}"
