"""A content digest for a parsed netlist - what actually identifies a design.

A results file that records only "benchmarks/adaptec1" records where a design was mounted, not
which design it was. That gets both directions wrong: two machines mounting the same benchmark at
different relative paths look incomparable, while editing a netlist in place - fixing a macro
size, regenerating from a different synthesis run - looks comparable, silently, forever.

So identity is taken from the parsed netlist rather than the path or the raw bytes. Parsed, not
bytes, on purpose: reformatting a Bookshelf file, reordering its net blocks or rewriting its
comments does not change the design, and should not invalidate a comparison; changing a macro's
width does, and must.

The digest is canonicalized so it depends only on the netlist's content, never on the incidental
ordering the parser happened to emit - macros sorted by name, each net's pins sorted by name, and
the nets themselves sorted. Placement ORDER is deliberately not folded in here: that is a
separate, independently-swappable axis (`OrderFn`), hashed separately by BenchmarkSpec, and
two runs differing only in order are running the same design.
"""
import hashlib
import json

from placax.types import Nets, SizeMap

_COORD_PRECISION = 6
"""Decimal places kept for sizes and pin offsets. Enough to distinguish any real geometry, while
keeping the digest stable against float formatting differences between parsers and platforms."""


def _round(value: float) -> float:
    return round(float(value), _COORD_PRECISION)


def canonical_netlist(macro_sizes: SizeMap, nets: Nets) -> dict:
    """The netlist reduced to a canonical, JSON-able form: sorted, rounded, order-independent."""
    macros = [
        [name, _round(width), _round(height)]
        for name, (width, height) in sorted(macro_sizes.items())
    ]
    # Sort pins within each net, then the nets themselves, so a parser that emits nets in file
    # order and one that emits them in any other order produce the same digest.
    canonical_nets = sorted(
        sorted([name, _round(x), _round(y)] for name, x, y in net)
        for net in nets
    )
    return {"macros": macros, "nets": canonical_nets}


def netlist_digest(macro_sizes: SizeMap, nets: Nets) -> str:
    """A short, stable content hash of a parsed netlist, for BenchmarkSpec.netlist_digest."""
    canonical = json.dumps(canonical_netlist(macro_sizes, nets), sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
