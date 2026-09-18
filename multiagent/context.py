"""One loaded design, ready for macro agents: warm start, bounds, footprints, neighbour table.

Everything every other module here needs, built once. The three decisions worth stating:

**The episode starts from the greedy-wiremask placement, not an empty canvas.** It is legal, and at
128 macros it is the best placement this project has - measured here at **441,257** real HPWL on
adaptec1 (224 grid, core canvas), against roughly 1.04M for PPO at a 200k-step budget. So the
agents' job is refinement, and the number to beat is the placement they were handed, which makes a
failure unambiguous instead of a matter of tuning. (An older comparison log quoted 406,631 for the
same agent; that run's directory is gone, so which canvas and order it used is unverified. Every
run here prints its own warm start, and that printed number is the one to compare against.)

**Positions stay in GRID units, as lower-left corners**, exactly like the core environment's
`positions`. They are float here rather than integer, because a continuous nudge is the only kind
of action an analytic gradient can flow through. Everything that scores a placement converts to
real-unit centers with `to_real_centers`, which is the core's own conversion.

**The defaults are the environment the shipped comparison preset runs in** - grid 224, the first
128 macros in alphabetical order, the design's core canvas. They are defaults rather than
requirements: every one is a flag, and two runs only mean something side by side if they agree on
all of them, which `compare.py` checks rather than trusts.
"""
import pathlib
from dataclasses import dataclass

from placax import _device  # noqa: F401  must precede jax imports
from placax.extras.legality import jitted_legality
from placax.types import EnvParams
from placax_agents.benchmark import Benchmark
from placax_agents.policy.scale import to_grid_units

import jax
import jax.numpy as jnp
import numpy as np

from multiagent import neighbors

DEFAULT_GRID = 224
"""MaskPlace's canvas resolution, and the one the shipped comparison runs used."""

DEFAULT_MACRO_BUDGET = 128
"""MaskPlace's --pnm default: the design's first 128 macros under the run's order."""


@dataclass(frozen=True)
class Context:
    """A design plus everything the agents, the baselines and the metrics need from it."""

    benchmark: Benchmark
    warm_start: jax.Array
    """`(n_macros, 2)` float32 grid lower-left corners - every macro placed, legally."""

    lo: jax.Array
    hi: jax.Array
    """Per-macro bounds on the lower-left corner: `hi = canvas - footprint`, so a macro clipped to
    these bounds is entirely ON the canvas. The continuous counterpart of `boundary_mask`, and the
    same trick the shipped continuous policy head uses - bound the body, not the corner."""

    sizes_grid: jax.Array
    """`(n_macros, 2)` footprints in grid cells, NOT rounded up. `to_grid_units` ceils to integer
    cells because a discrete placement occupies whole cells; a continuous one does not, and
    rounding the footprint up here would shrink the legal region by up to a cell per macro."""

    neighbor_idx: jax.Array
    neighbor_weight: jax.Array
    neighbor_valid: jax.Array
    """`(n_macros, k)` top-k wired partners, their normalized clique weight, and whether the slot
    is a real connection - see neighbors.py."""

    connection_weights: np.ndarray
    """The full `(n, n)` weight matrix, kept for run summaries and for grouping experiments later."""

    @property
    def params(self) -> EnvParams:
        return self.benchmark.params

    @property
    def n_macros(self) -> int:
        return self.benchmark.params.n_macros

    @property
    def canvas(self) -> jax.Array:
        return jnp.array(
            [self.params.grid_x, self.params.effective_grid_y], dtype=jnp.float32
        )

    def legality_of(self, positions: jax.Array) -> dict:
        """The core's own legality measurement, on the placement as it would be EXPORTED.

        Rounded, for the reason `experiment.run.score` gives: two macros at 3.4 and 3.6 overlap in
        the file a tool receives, whatever the float coordinates say.
        """
        grid_sizes = to_grid_units(self.benchmark.sizes_array, self.benchmark.cell_size)
        return jitted_legality(
            jnp.round(positions).astype(jnp.int32), grid_sizes, self.params
        ).to_dict()


def build(
    benchmark_dir: str | pathlib.Path,
    grid: int = DEFAULT_GRID,
    macro_budget: int | None = DEFAULT_MACRO_BUDGET,
    canvas: str = "core",
    k_neighbors: int = 4,
) -> Context:
    """Loads a design, places every macro greedily, and precomputes the rest."""
    benchmark_dir = pathlib.Path(benchmark_dir)
    benchmark = Benchmark.load(
        benchmark_dir, grid=grid, macro_budget=macro_budget, canvas=canvas
    )

    # 1. The warm start: the greedy-wiremask heuristic, every macro placed. Imported here rather
    #    than at module scope because the agent module pulls in the whole training stack.
    from placax_agents.agents.baselines import GreedyWiremaskAgent

    warm_start = GreedyWiremaskAgent(benchmark).best_positions({}).astype(jnp.float32)

    # 2. Bounds. A macro wider than the canvas would give hi < lo, which clip cannot satisfy -
    #    clamp at 0 so it sits at the corner instead of inverting the interval.
    sizes_grid = benchmark.sizes_array / benchmark.cell_size
    canvas_extent = jnp.array(
        [benchmark.params.grid_x, benchmark.params.effective_grid_y], dtype=jnp.float32
    )
    hi = jnp.clip(canvas_extent - sizes_grid, 0.0, None)
    lo = jnp.zeros_like(hi)

    # 3. Who is wired to whom.
    weights = neighbors.connection_weights(
        benchmark.nets, benchmark.name_to_idx, benchmark.params.n_macros
    )
    idx, weight, valid = neighbors.top_k_neighbors(weights, k_neighbors)

    return Context(
        benchmark=benchmark,
        warm_start=jnp.clip(warm_start, lo, hi),
        lo=lo,
        hi=hi,
        sizes_grid=sizes_grid,
        neighbor_idx=jnp.asarray(idx),
        neighbor_weight=jnp.asarray(weight),
        neighbor_valid=jnp.asarray(valid),
        connection_weights=weights,
    )
