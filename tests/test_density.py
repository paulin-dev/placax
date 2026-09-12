"""The differentiable density term - the piece SHAC was blocked on, checked against hand arithmetic.

`docs/Action_Space_Decision.md` measured the wirelength half of SHAC's problem and left this half
named as "the substantial piece of work". A term that is differentiable in principle and wrong in
its arithmetic would be worse than no term at all, so the area computation is pinned against
overlaps small enough to work out by hand, and the gradient is pinned against the two properties
that decide whether an optimizer can use it: it must be nonzero where the cost is active, and it
must point the right way.

The measurement that motivated the default lives in the decision record; what is asserted here is
the behaviour that measurement relies on.
"""
import pytest

from placax.extras.density import (  # noqa: F401  must precede jax imports
    area_density, density_overflow, gradient_coverage, make_density_cost, out_of_bounds_cost,
)
from placax.types import EnvParams

import jax
import jax.numpy as jnp

PARAMS = EnvParams(grid=4, n_macros=1)


def _centers(*pairs):
    return jnp.array(pairs, dtype=jnp.float32)


# --------------------------------------------------------------- the area arithmetic


def test_a_macro_filling_one_bin_gives_that_bin_exactly_one() -> None:
    # A 1x1 macro centered in bin (1, 1) covers it completely and nothing else.
    density = area_density(_centers((1.5, 1.5)), _centers((1.0, 1.0)), PARAMS, cell_size=1.0)
    assert float(density[1, 1]) == pytest.approx(1.0)
    assert float(density.sum()) == pytest.approx(1.0)


def test_a_macro_straddling_two_bins_splits_its_area_between_them() -> None:
    # Centered on the boundary between bins 1 and 2: half its area falls in each.
    density = area_density(_centers((2.0, 1.5)), _centers((1.0, 1.0)), PARAMS, cell_size=1.0)
    assert float(density[1, 1]) == pytest.approx(0.5)
    assert float(density[2, 1]) == pytest.approx(0.5)


def test_total_density_equals_total_macro_area_in_bin_units() -> None:
    # Area is conserved: whatever the placement, the map sums to the macros' area in bins. This
    # is the property that makes "overflow" mean something - and, read the other way, the reason
    # a penalty active on EVERY bin has no gradient, since moving a macro cannot change the total.
    sizes = _centers((2.0, 1.0), (1.0, 3.0))
    density = area_density(_centers((1.2, 2.3), (2.7, 1.6)), sizes, PARAMS, cell_size=1.0)
    expected = float((sizes[:, 0] * sizes[:, 1]).sum())
    assert float(density.sum()) == pytest.approx(expected, rel=1e-5)


def test_two_macros_in_one_bin_stack_above_full() -> None:
    # Overlap is what the term exists to see: two 1x1 macros on the same bin make it 2.0.
    density = area_density(
        _centers((1.5, 1.5), (1.5, 1.5)), _centers((1.0, 1.0), (1.0, 1.0)), PARAMS, cell_size=1.0
    )
    assert float(density[1, 1]) == pytest.approx(2.0)
    assert float(density_overflow(density, target_density=1.0)) == pytest.approx(1.0)


def test_an_unplaced_macro_contributes_nothing() -> None:
    density = area_density(
        _centers((1.5, 1.5), (2.5, 2.5)), _centers((1.0, 1.0), (1.0, 1.0)), PARAMS,
        cell_size=1.0, placed_mask=jnp.array([True, False]),
    )
    assert float(density.sum()) == pytest.approx(1.0)


def test_cell_size_converts_real_units_to_bins() -> None:
    # Positions are REAL units, like every other reward input; a 2.0-unit macro on a 2.0-unit
    # cell size is exactly one bin.
    density = area_density(_centers((3.0, 3.0)), _centers((2.0, 2.0)), PARAMS, cell_size=2.0)
    assert float(density[1, 1]) == pytest.approx(1.0)


# --------------------------------------------------------------- the penalty


def test_a_placement_inside_the_target_costs_nothing() -> None:
    # The defining property of a penalty, and its documented limit: silent when not violated.
    density = area_density(_centers((0.5, 0.5)), _centers((1.0, 1.0)), PARAMS, cell_size=1.0)
    assert float(density_overflow(density, target_density=1.0)) == 0.0


def test_lowering_the_target_turns_separation_into_spreading() -> None:
    """`target_density` is the knob between "don't overlap" and "spread out".

    At 1.0 a legal placement is free, so a continuous method gets no signal until macros collide.
    Below the design's own average density the term is active on a merely-crowded region, which
    is what gives an analytic-gradient method something to descend before anything overlaps.
    """
    centers, sizes = _centers((1.5, 1.5)), _centers((1.0, 1.0))
    density = area_density(centers, sizes, PARAMS, cell_size=1.0)
    assert float(density_overflow(density, target_density=1.0)) == 0.0
    assert float(density_overflow(density, target_density=0.5)) > 0.0


def test_leaving_the_canvas_is_charged_by_area_and_by_distance() -> None:
    """Both halves, and the area one is what stops a policy escaping instead of packing.

    The area term is in bin units, the same units `density_overflow` charges in, so a macro that
    leaves pays what a macro that overlaps pays. The distance term is what keeps a gradient once
    the macro is ENTIRELY outside and its escaped area has stopped growing.
    """
    sizes = _centers((1.0, 1.0))
    half_out = float(out_of_bounds_cost(_centers((0.0, 0.5)), sizes, PARAMS, cell_size=1.0))
    fully_out = float(out_of_bounds_cost(_centers((-0.5, 0.5)), sizes, PARAMS, cell_size=1.0))
    further = float(out_of_bounds_cost(_centers((-2.0, 0.5)), sizes, PARAMS, cell_size=1.0))

    assert half_out > 0.0
    assert fully_out > half_out
    # Past the edge the escaped AREA saturates at the macro's own area, so without the distance
    # term this would be flat and a stranded macro would never be pulled back.
    assert further > fully_out


def test_escaping_the_canvas_is_never_cheaper_than_overlapping() -> None:
    """The arbitrage a real SHAC run found, before the area term existed.

    With out-of-bounds charged only as squared overhang distance, a continuous policy on a crowded
    canvas learned that stepping off the edge (cost ~1) beat overlapping a 2x4 macro (cost ~8).
    Every density weight tried produced 0% overlap and 100% out-of-bounds: a perfectly legal
    placement of an empty canvas. A macro fully outside must cost at least what the same macro
    fully overlapping costs, or the penalty pays for the escape.
    """
    sizes = _centers((2.0, 2.0))
    overlapping = float(out_of_bounds_cost(  # fully on canvas, so bounds cost nothing
        _centers((2.0, 2.0)), sizes, PARAMS, cell_size=1.0
    ))
    assert overlapping == 0.0

    stacked = area_density(
        _centers((2.0, 2.0), (2.0, 2.0)), jnp.concatenate([sizes, sizes]), PARAMS, cell_size=1.0
    )
    overlap_cost = float(density_overflow(stacked, target_density=1.0))
    escaped_cost = float(out_of_bounds_cost(_centers((-1.0, 2.0)), sizes, PARAMS, cell_size=1.0))
    assert escaped_cost >= overlap_cost


def test_a_macro_inside_the_canvas_is_not_charged_for_bounds() -> None:
    assert float(
        out_of_bounds_cost(_centers((2.0, 2.0)), _centers((1.0, 1.0)), PARAMS, cell_size=1.0)
    ) == 0.0


# --------------------------------------------------------------- the gradient, which is the point


def test_area_flows_from_a_congested_bin_toward_a_free_one() -> None:
    """Not just "a gradient exists" - it has to point somewhere useful.

    This is a CONGESTION gradient, not a pairwise repulsion, and the difference is worth having a
    test about. `B` fills bin 1 exactly; `A` straddles bin 1 (now over target) and bin 2 (empty).
    The cost falls as A's area moves into bin 2, so A is pushed off B - but it is pushed there
    because bin 2 has room, not because B is next to it. Where the free space lies on the far side
    of a neighbour, this term will push a macro TOWARD that neighbour, and that is correct
    behaviour for a density penalty rather than a bug to be found later.
    """
    sizes = _centers((1.0, 1.0), (1.0, 1.0))
    cost = make_density_cost(sizes, PARAMS, cell_size=1.0, target_density=1.0)
    #        A straddles bins 1|2       B fills bin 1
    centers = _centers((1.9, 1.5), (1.5, 1.5))
    gradient = jax.grad(lambda xy: cost(xy, None))(centers)
    # d(cost)/dx < 0 for A means the cost falls as A moves right, into the free bin.
    assert float(gradient[0, 0]) < 0.0


def test_a_macro_off_the_canvas_is_pulled_back_in() -> None:
    sizes = _centers((1.0, 1.0))
    cost = make_density_cost(sizes, PARAMS, cell_size=1.0)
    gradient = jax.grad(lambda xy: cost(xy, None))(_centers((-1.0, 2.0)))
    assert float(gradient[0, 0]) < 0.0, "cost must fall as the macro moves back onto the canvas"


def test_gradient_coverage_counts_macros_that_actually_receive_one() -> None:
    sizes = _centers((1.0, 1.0), (1.0, 1.0))
    cost = make_density_cost(sizes, PARAMS, cell_size=1.0, target_density=1.0)
    overlapping, considered = gradient_coverage(cost, _centers((1.7, 1.5), (2.0, 1.5)))
    assert considered == 2 and overlapping == 2

    # Far apart and comfortably inside the canvas: nothing is violated, so nothing receives a
    # gradient. That is the penalty's documented silence, asserted rather than left to be
    # discovered in training. Kept off the canvas edge deliberately - a macro flush against it
    # sits exactly on the hinge, where JAX's tie subgradient (0.5) reports a gradient that an
    # optimizer cannot use, the same trap the decision record's gamma measurement fell into.
    apart, _considered = gradient_coverage(cost, _centers((0.6, 0.6), (2.6, 2.6)))
    assert apart == 0


def test_a_uniformly_violated_region_has_no_gradient_left_to_give() -> None:
    """The non-obvious property, and the reason the measured coverage curve is not monotonic.

    Density is conserved area, so if EVERY bin is already above the target, moving a macro shifts
    area between bins that are all charged at the same rate and the total overflow does not
    change. The gradient lives on the frontier between over-target and under-target bins, not
    inside the congested region - so driving `target_density` very low buys less signal, not more.
    """
    sizes = _centers((4.0, 4.0))          # covers the whole 4x4 canvas
    cost = make_density_cost(sizes, PARAMS, cell_size=1.0, target_density=0.01)
    gradient = jax.grad(lambda xy: cost(xy, None))(_centers((2.0, 2.0)))
    assert float(jnp.abs(gradient).sum()) == pytest.approx(0.0, abs=1e-6)
