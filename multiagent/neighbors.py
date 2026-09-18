"""Which macros are wired to which - the one piece of netlist structure a macro-agent can see.

A macro agent that sees only itself has no way to know which direction shortens a wire: wirelength
is a property of PAIRS. So the `m1` view hands each macro its strongest partners, and this module
decides what "strongest" means.

**The weight is the clique model, which is the standard one.** A net with `m` pins on distinct
macros is turned into all pairs among those macros, each weighted `2 / m`. The division is what
keeps a 200-pin clock net from swamping every real connection: a two-pin net contributes 1.0 to
its single pair, while a 50-pin net contributes 0.04 to each of its 1225 pairs. This is the same
weighting a hypergraph partitioner uses for the same reason, and it is computed once from the
netlist rather than learned.

Everything here is NumPy, on purpose. It runs once per run, before any JAX tracing, and the
result is a small fixed-shape integer table - so it never has to be jittable.
"""
import numpy as np

from placax.types import Nets


def connection_weights(nets: Nets, name_to_idx: dict[str, int], n_macros: int) -> np.ndarray:
    """`(n_macros, n_macros)` symmetric clique weights, zero on the diagonal.

    Pins naming something that is not one of this run's macros are skipped: a macro budget keeps
    only the first N macros, and the nets still mention the ones it dropped.
    """
    weights = np.zeros((n_macros, n_macros), dtype=np.float32)
    for net in nets:
        # 1. Distinct macros this net actually touches, in THIS run's index space.
        touched = {name_to_idx[name] for name, _dx, _dy in net if name in name_to_idx}
        if len(touched) < 2:
            # A net with one macro (or none) says nothing about any pair.
            continue
        # 2. Clique weight: every pair on the net, discounted by how many pins the net has.
        members = np.fromiter(sorted(touched), dtype=np.int32)
        weight = 2.0 / len(members)
        rows, cols = np.meshgrid(members, members, indexing="ij")
        weights[rows, cols] += weight
    np.fill_diagonal(weights, 0.0)
    return weights


def top_k_neighbors(weights: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each macro's `k` most strongly connected partners: `(idx, weight, valid)`.

    `weight` is normalized by the largest weight anywhere in the design, so the feature a policy
    sees lands in [0, 1] on every benchmark rather than in whatever units this netlist's fanout
    happens to produce. `valid` is False where a macro has fewer than `k` partners at all - it is
    a feature the policy reads, not a filter, because the number of neighbours has to be a static
    shape and a macro connected to nothing still has to produce an action.
    """
    n_macros = weights.shape[0]
    k = min(k, max(1, n_macros - 1))
    # argpartition puts the k largest anywhere in the first k slots; sort those k so the strongest
    # partner is always feature slot 0 rather than wherever the partition left it.
    partitioned = np.argpartition(-weights, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(n_macros)[:, None]
    order = np.argsort(-weights[rows, partitioned], axis=1)
    idx = partitioned[rows, order].astype(np.int32)
    picked = weights[rows, idx]
    scale = float(weights.max()) or 1.0
    return idx, (picked / scale).astype(np.float32), picked > 0.0


def degree(weights: np.ndarray) -> np.ndarray:
    """Total connection weight per macro - "how wired in is this block", for run summaries."""
    return weights.sum(axis=1)
