"""Named builders for every swappable component, so a config can name one in JSON.

An ExperimentConfig has to survive a round trip through a results file, which means it can hold
registry keys and JSON scalars but not live Python objects. This module is the lookup that turns
`Spec("wiremask_quality", {"margin": 1.0})` back into the actual function.

Adding a component means adding one entry. Nothing else in the experiment machinery needs to know
it exists, and every script that consumes configs gains it at once - which is the whole point of
routing both training scripts through one registry instead of letting each hardcode its own
choices.

**These registries are OPEN, and the tables below are defaults rather than a closed set.** The
project's rule is that anything a different team might do differently is a parameter, not a
hard-coded call - so a reward, a state representation, a policy, an agent or an action space of
your own is an ordinary Python function you `register()` from your own code. Nothing in this file
needs editing. See `register` at the bottom, and `docs/Reference.md` for the two routes: register
a name (so a config can select it and round trip through JSON), or hand a live object straight to
`build()`/`run_experiment()` for a one-off that never needs to be written down.
"""
from placax.extras.masks import regularity_cost, regularity_max  # noqa: F401  must precede jax imports
from placax.log import Log
from placax.netlist.order import alphabetical_order, area_desc_order, connectivity_order_for
from placax.netlist.padding import build_macro_net_index
from placax.types import EnvParams, OrderFn
from placax_agents.policy.action import make_wiremask_quality_illegal
from placax_agents.policy.architectures.cnn import CNNActorCritic
from placax_agents.policy.observation import make_wiremask_observation, observation
from placax_agents.training.algorithm.loss import huber_value_loss, mse_value_loss
from placax_agents.training.algorithm.split_optimizer import make_grouped_optimizer
from placax_agents.training.reward import (
    make_differentiable_reward, make_expert_reward, make_hpwl_congestion_reward,
    make_scaled_hpwl_reward, make_scaled_smoothed_reward,
)

import functools

import optax

# ---------------------------------------------------------------------------
# Macro placement order.  (benchmark_name, **kwargs) -> OrderFn
# ---------------------------------------------------------------------------

MASKPLACE_CONNECTIVITY_WEIGHTS = {"ariane": (30000.0, 1000.0), "bigblue3": (1.0, 100000.0)}
"""MaskPlace's own per-benchmark overrides for `candidates*W1 + degree*W2 + area`, from its
get_node_id_to_name_topology. Matched by substring, so "ariane133" picks up the "ariane" entry."""

DEFAULT_CONNECTIVITY_WEIGHTS = (1.0, 1000.0)


def _maskplace_connectivity_weights(benchmark_name: str) -> tuple[float, float]:
    """(candidate_weight, degree_weight) for a benchmark, per MaskPlace's own per-design overrides."""
    name = benchmark_name.lower()
    for key, weights in MASKPLACE_CONNECTIVITY_WEIGHTS.items():
        if key in name:
            return weights
    return DEFAULT_CONNECTIVITY_WEIGHTS


def _order_alphabetical(_benchmark_name: str) -> OrderFn:
    return alphabetical_order


def _order_area_desc(_benchmark_name: str) -> OrderFn:
    return area_desc_order


def _order_connectivity(
    _benchmark_name: str, candidate_weight: float = 1.0, degree_weight: float = 1000.0
) -> OrderFn:
    return connectivity_order_for(candidate_weight, degree_weight)


def _order_connectivity_maskplace(benchmark_name: str) -> OrderFn:
    """connectivity_order with MaskPlace's own weights for this specific benchmark."""
    return connectivity_order_for(*_maskplace_connectivity_weights(benchmark_name))


ORDERS = {
    "alphabetical": _order_alphabetical,
    "area_desc": _order_area_desc,
    "connectivity": _order_connectivity,
    "connectivity_maskplace": _order_connectivity_maskplace,
}

# ---------------------------------------------------------------------------
# Reward.  (grid, **kwargs) -> (pin_idx, pin_offset, valid_mask, sizes, cell_size) -> RewardFn
# ---------------------------------------------------------------------------

MASKPLACE_REWARD_DIVISOR = 200.0
"""MaskPlace's own constant divisor on its (grid-unit) reward, applied per step in PPO2.py."""


def _reward_hpwl(_grid: int, dense: bool = False, reward_scale: float = 1.0):
    """Plain -HPWL in real units, sparse (terminal) by default."""
    return functools.partial(make_scaled_hpwl_reward, dense=dense, reward_scale=reward_scale)


def _reward_maskplace(
    grid: int, regularity_weight: float = 0.0, regularity_mode: str = "corner",
    reward_divisor: float = MASKPLACE_REWARD_DIVISOR,
):
    """MaskPlace's dense per-step reward: the real-unit HPWL delta expressed back in grid units
    and divided by 200, optionally plus EXPlace's regularity (periphery) term.

    regularity_weight=0.0 is exactly MaskPlace's reward. Above 0 it adds the one EXPlace term
    that needs none of the preprocessed clustering/dataflow data EXPlace ships separately.
    """

    def factory(padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size):
        # Benchmark.load builds its own EnvParams after this factory runs, so rebuild the
        # matching one here from the grid and the macro count sizes_array already carries.
        params = EnvParams(grid=grid, n_macros=sizes_array.shape[0])
        return make_expert_reward(
            padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size, params,
            dense=True, reward_scale=1.0 / (cell_size * reward_divisor),
            regularity_weight=regularity_weight, regularity_mode=regularity_mode,
        )

    return factory


SMOOTHED_GAMMA_CELLS = 1.0
"""Default log-sum-exp smoothing, in GRID CELLS - see `_reward_smoothed` for the measurement."""


def _reward_smoothed(_grid: int, dense: bool = False, reward_scale: float = 1.0,
                     gamma_cells: float = SMOOTHED_GAMMA_CELLS):
    """-WAWL (log-sum-exp smoothed wirelength) in real units - HPWL's differentiable surrogate.

    **gamma is in grid cells, not in design units, and that is load-bearing.** log-sum-exp
    replaces each `max` with `gamma * log(sum(exp(x / gamma)))`, so gamma has to be commensurate
    with the coordinates. A design's coordinates run to ~1e4, and an absolute `gamma=1.0` makes
    `exp(x / gamma)` saturate in float32 - the surrogate degenerates back to a hard max, and the
    gradient gets *worse* than raw HPWL's. Measured on adaptec1, 543 macros, 514 connected, as
    the fraction of connected macros receiving a nonzero d(-WAWL)/d(position):

        gamma (cells)   dense    fidelity vs true HPWL
        raw HPWL        42.2%    exact
        0.006  (=1.0)    1.4%    +0.00%     <- the old absolute default, degenerate
        0.166           57.2%    +0.08%
        0.552           98.8%    +0.33%
        1.0            100.0%    +1.00%
        2.8            100.0%    +4.96%

    One grid cell buys complete gradient coverage for one percent of fidelity, which is what
    settles docs/Action_Space_Decision.md's option D: the sparse-gradient half of SHAC's problem
    is solved and cheap. The remaining half - a differentiable density/overlap term - is not.
    """
    def factory(padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size):
        return make_scaled_smoothed_reward(
            padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size,
            dense=dense, reward_scale=reward_scale, gamma=gamma_cells * cell_size,
        )

    return factory


def _reward_hpwl_congestion(grid: int, congestion_weight: float = 1.0, capacity: float = 1.0,
                            dense: bool = False, reward_scale: float = 1.0):
    """-(HPWL + w * RUDY overflow) - wirelength traded against routing congestion.

    The second objective docs §12's reward comparison asks for ("HPWL vs. HPWL+congestion, agent
    held fixed"), which until now could not be selected from a config at all. Size
    congestion_weight against the HPWL term's real magnitude with scripts/measure_reward_terms.py.
    """

    def factory(padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size):
        # Benchmark.load builds its own EnvParams after this factory runs, so rebuild the
        # matching one here - the same pattern _reward_maskplace uses.
        params = EnvParams(grid=grid, n_macros=sizes_array.shape[0])
        return make_hpwl_congestion_reward(
            padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size, params,
            congestion_weight=congestion_weight, capacity=capacity, dense=dense,
            reward_scale=reward_scale,
        )

    return factory


def _reward_differentiable(
    grid: int, density_weight: float = 1.0, target_density: float = 1.0,
    bounds_weight: float = 1.0, gamma_cells: float = SMOOTHED_GAMMA_CELLS,
    dense: bool = True, reward_scale: float = 1.0,
):
    """-(smoothed wirelength + density overflow + out-of-bounds): the objective SHAC descends.

    The reward an analytic-gradient method needs, and the only one here whose LEGALITY has a
    gradient. Under the discrete spaces legality is a mask and overlap is invisible to `jax.grad`;
    a continuous action cannot be masked, so this charges for overlap and for leaving the canvas
    instead (`placax/extras/density.py`).

    `target_density` is the knob between "don't overlap" (1.0 - silent on any legal placement) and
    "spread out" (below the design's own average density - active on a merely crowded region).
    Measured on adaptec1, whose macros cover 47.7% of the canvas, as the share of macros receiving
    a nonzero gradient from a uniformly random placement:

        target_density   overflow   gradient coverage
        1.0               3804       57.3%
        0.9               5362       63.2%
        0.8               6996       70.7%
        0.7               8713       78.6%     <- the frontier is richest here
        0.5              12376       67.4%
        0.3              16352       53.4%

    Coverage falls again below 0.7 because density is conserved area: once every bin is over
    target, moving a macro shifts area between bins charged at the same rate and the total does
    not move. The gradient lives on the frontier between over- and under-target bins, so a target
    under the design's own density erases the very thing it was lowered to create.
    """

    def factory(padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size):
        # Benchmark.load builds its own EnvParams after this factory runs, so rebuild the matching
        # one here - the same pattern _reward_maskplace uses.
        params = EnvParams(grid=grid, n_macros=sizes_array.shape[0])
        return make_differentiable_reward(
            padded_pin_idx, padded_pin_offset, valid_mask, sizes_array, cell_size, params,
            density_weight=density_weight, target_density=target_density,
            bounds_weight=bounds_weight, gamma=gamma_cells * cell_size,
            dense=dense, reward_scale=reward_scale,
        )

    return factory


REWARDS = {
    "differentiable": _reward_differentiable,
    "hpwl": _reward_hpwl,
    "maskplace": _reward_maskplace,
    "smoothed": _reward_smoothed,
    "hpwl_congestion": _reward_hpwl_congestion,
}

# ---------------------------------------------------------------------------
# Observation (state representation).  (benchmark, **kwargs) -> StateFn
# ---------------------------------------------------------------------------


def _state_canvas(benchmark, lookahead: int = 1):
    """The bare canvas observation, bound to this benchmark's real cell_size."""
    state_fn = functools.partial(observation, cell_size=benchmark.cell_size, lookahead=lookahead)
    # Carried through the partial: what an observation NEEDS from the action space is a property
    # of the observation, and build() checks it. See placax/action_space.py's target().
    state_fn.needs_current_macro = getattr(observation, "needs_current_macro", True)
    return state_fn


def _state_wiremask(benchmark, lookahead: int = 2):
    """Canvas plus per-cell HPWL-increase previews for the next `lookahead` macros."""
    # Precompute, once, which nets touch each macro, so it isn't redone on every step.
    macro_net_idx, macro_net_offset, macro_net_valid = build_macro_net_index(
        benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
        n_macros=benchmark.params.n_macros,
    )
    return make_wiremask_observation(
        benchmark.padded_pin_idx, benchmark.padded_pin_offset, benchmark.valid_mask,
        macro_net_idx, macro_net_offset, macro_net_valid,
        cell_size=benchmark.cell_size, lookahead=lookahead,
    )


STATES = {"canvas": _state_canvas, "wiremask": _state_wiremask}

# ---------------------------------------------------------------------------
# Extra action-legality masks.  (benchmark, **kwargs) -> ExtraIllegalFn
# ---------------------------------------------------------------------------


def _mask_wiremask_quality(benchmark, margin: float = 1.0):
    """MaskPlace's soft-coefficient rule: rule out cells whose normalized wiremask exceeds the
    legal minimum plus `margin`."""
    return make_wiremask_quality_illegal(margin=margin, cell_size=benchmark.cell_size)


MASKS = {"wiremask_quality": _mask_wiremask_quality}

# ---------------------------------------------------------------------------
# Initial placement (warm start).  (benchmark, **kwargs) -> InitFn
# ---------------------------------------------------------------------------
#
# TILOS found Circuit Training leans heavily on the initial placement handed to it by a
# commercial tool - removing it measurably worsened routed wirelength. So "what did this run
# start from" is a result-changing axis, and until now it was neither selectable nor recorded:
# every run silently started from an empty canvas with nothing saying so. `empty` is that
# behavior, named, so today's runs are on the record as having used it.
#
# An InitFn is resolved ONCE per run, not per episode: the warm start is a property of the
# environment, and the number of macros the agent still has to place has to be a static shape.


def _init_empty(_benchmark):
    """Nothing pre-placed - the historical behavior, now written down rather than assumed."""
    return lambda _key: None


def _init_greedy_wiremask_prefix(benchmark, n_macros: int | None = 8):
    """Pre-place the first `n_macros` macros with the greedy-wiremask heuristic.

    A free, open-source warm start built entirely from shipped pieces, which is exactly the
    question docs §5.3 leaves open: can one recover the benefit Circuit Training gets from a
    commercial initial placement? The agent then starts from a partly-populated canvas and
    places only what is left.

    `n_macros=None` places EVERY macro, which is not a prefix at all but a complete starting
    placement - what a perturbation space has to begin from, since it moves macros rather than
    appending them. Spelled as None rather than as a number someone has to keep in step with the
    design's macro count (543 here, 128 under MaskPlace's budget, different on every benchmark).
    """
    from placax_agents.agents.baselines import GreedyWiremaskAgent

    def init_fn(_key):
        # The heuristic is deterministic, so the key is unused and the warm start is a pure
        # function of the netlist and the placement order.
        placement = GreedyWiremaskAgent(benchmark).best_positions({})
        if n_macros is None:
            return placement
        # Keep only the prefix; everything after it returns to the unplaced sentinel.
        return placement.at[n_macros:].set(-1)

    if n_macros is not None and n_macros <= 0:
        raise ValueError(
            f"greedy_wiremask_prefix needs n_macros > 0, or None for every macro, got {n_macros}"
        )
    return init_fn


INITS = {"empty": _init_empty, "greedy_wiremask_prefix": _init_greedy_wiremask_prefix}

# ---------------------------------------------------------------------------
# Legalization.  (benchmark, **kwargs) -> LegalizeFn
# ---------------------------------------------------------------------------
#
# A LegalizeFn takes {macro_name: (x, y)} REAL-unit lower-left corners plus the macro sizes, and
# returns the same shape made physically realizable. It runs at the boundary between the
# environment and the physical world (`experiment.export`), never inside the episode: it moves
# already-placed macros, which the sequential-constructive kernel has no action for, and putting
# it in the loop would change what the reward is computed over. See docs/Action_Space_Decision.md
# - a perturbation action space is what would let a legalizer participate in the episode proper.
#
# Measured before being written, on adaptec1: the grid's rows land on a real placement row 0 times
# out of 224, but the worst-case snap is pitch/2 = 6 units = 0.116 of a grid cell. So this is a
# correctness fix with a negligible wirelength cost, which is worth knowing rather than assuming
# in either direction.


def _legalizer_row_snap(benchmark, clamp_to_core: bool = True):
    """Snaps every macro to the nearest legal site and row, keeping it inside the core area."""
    rows = benchmark.rows
    if rows is None:
        raise ValueError(
            "the row_snap legalizer needs the design's placement rows, and this benchmark "
            "carries none (a protobuf netlist, or a Bookshelf directory with no .scl). Legality "
            "against rows cannot be defined for it, so leave EnvironmentSpec.legalization at None."
        )

    def legalize(placement: dict, macro_sizes: dict) -> dict:
        legalized = {}
        for name, (x, y) in placement.items():
            width, height = macro_sizes.get(name, (0.0, 0.0))
            if clamp_to_core:
                legalized[name] = rows.snap(x, y, width, height)
            else:
                # Snap only - a macro outside the core stays outside, and is reported as illegal
                # rather than silently dragged in.
                legalized[name] = rows.snap(x, y)
        return legalized

    return legalize


LEGALIZERS = {"row_snap": _legalizer_row_snap}

# ---------------------------------------------------------------------------
# Action space.  (benchmark, **kwargs) -> ActionSpace
# ---------------------------------------------------------------------------
#
# The last thing in the kernel that was not swappable. `discrete_grid` is the historical
# behaviour and stays the default, so a config that does not mention it runs exactly as before.


def _action_space_discrete_grid(_benchmark):
    """One macro per step, in array order, at an integer grid cell."""
    from placax.action_space import DiscreteGridPlacement

    return DiscreteGridPlacement()


def _action_space_perturbation(_benchmark, n_moves: int = 64):
    """Move an already-placed macro: the action is (macro, x, y).

    Needs a complete initial placement, since there is nothing to move otherwise - pair it with an
    `initial_placement` that fills the canvas.
    """
    from placax.action_space import Perturbation

    return Perturbation(n_moves=n_moves)


def _action_space_oriented_grid(_benchmark):
    """Constructive, but the action is (x, y, orientation) - the agent chooses the turn too."""
    from placax.action_space import OrientedGridPlacement

    return OrientedGridPlacement()


def _action_space_continuous(_benchmark):
    """One macro per step at a REAL-VALUED coordinate - the space an analytic gradient needs.

    Pair it with the `differentiable` reward and no action mask: legality cannot be masked out of
    a continuous distribution, so it has to be charged for instead. `build()` refuses the other
    combinations rather than letting a mask be silently ignored.
    """
    from placax.action_space import ContinuousPlacement

    return ContinuousPlacement()


ACTION_SPACES = {
    "continuous": _action_space_continuous,
    "discrete_grid": _action_space_discrete_grid,
    "oriented_grid": _action_space_oriented_grid,
    "perturbation": _action_space_perturbation,
}

# ---------------------------------------------------------------------------
# Policy architecture.  (benchmark, **kwargs) -> nn.Module
# ---------------------------------------------------------------------------


def _resnet_backbone(pretrained: bool = True):
    """Real ImageNet weights when placax[resnet] is installed, else a same-shape offline stand-in."""
    from placax_agents.policy.architectures.resnet_cnn import (
        build_pretrained_resnet_backbone,
        build_untrained_resnet_backbone,
    )

    if not pretrained:
        return build_untrained_resnet_backbone()
    try:
        import flaxmodels  # noqa: F401
    except ImportError:
        Log.info("flaxmodels not installed (pip install placax[resnet]) - using an offline, "
                 "untrained ResNet backbone instead of real ImageNet weights.")
        return build_untrained_resnet_backbone()
    return build_pretrained_resnet_backbone()


def _policy_cnn(_benchmark, features: int = 16, num_conv_layers: int = 2):
    return CNNActorCritic(features=features, num_conv_layers=num_conv_layers)


def _policy_mlp(benchmark, features: int = 256, num_layers: int = 2):
    """Reads raw coordinates rather than the canvas image - §12's non-CNN arm.

    The state-representation study runs on THIS axis rather than on `state`: `observation()`
    returns both an image and the coordinates, and the architecture picks which it consumes. Two
    arms therefore share an environment exactly and differ only in `agent.policy`, which is a
    stronger comparison than the spec assumed - it holds at environment_hash, not just task_hash.
    """
    from placax_agents.policy.architectures.mlp import MLPActorCritic

    return MLPActorCritic(
        grid_x=benchmark.params.grid_x, grid_y=benchmark.params.effective_grid_y,
        size_scale=float(benchmark.sizes_array.max()), features=features, num_layers=num_layers,
    )


def _policy_wiremask_cnn(_benchmark, features: int = 16, num_conv_layers: int = 2):
    """The plain CNN plus a wiremask input channel. Requires the `wiremask` state representation.

    Shipped and documented since v4 but unregistered until now, which meant an architecture the
    spec describes could not be selected from a config at all - the one place a component being
    "available" has to mean something.
    """
    from placax_agents.policy.architectures.wiremask_cnn import WiremaskCNNActorCritic

    return WiremaskCNNActorCritic(features=features, num_conv_layers=num_conv_layers)


def _policy_resnet_coarse_fine(benchmark, critic_style: str = "step_embedding", pretrained: bool = True):
    """MaskPlace's own network shape: fine + coarse-ResNet branches, step-embedding critic."""
    from placax_agents.policy.architectures.resnet_cnn import ResNetCoarseFineActorCritic

    return ResNetCoarseFineActorCritic(
        resnet_backbone=_resnet_backbone(pretrained), params=benchmark.params,
        cell_size=benchmark.cell_size, critic_style=critic_style,
    )


def _policy_oriented_cnn(benchmark, features: int = 16, num_conv_layers: int = 2):
    """The plain CNN plus a head that chooses each macro's quarter turn - PPO's `oriented_grid` arm.

    The architecture that makes orientation learnable rather than only searchable: it emits
    `(grid_x, grid_y, 4)` logits, declares `oriented_grid` as the space it can drive, and gets
    per-turn legality from `policy.action.oriented_illegal_actions`. Pair it with
    `EnvironmentSpec.action_space = Spec("oriented_grid")` and a reward that sees orientations -
    every shipped one does.
    """
    from placax_agents.policy.architectures.oriented_cnn import OrientedCNNActorCritic

    return OrientedCNNActorCritic(
        features=features, num_conv_layers=num_conv_layers,
        size_scale=float(benchmark.sizes_array.max()),
    )


def _policy_continuous(benchmark, features: int = 256, num_layers: int = 2,
                       init_log_std: float = -1.0):
    """Mean + log-std over a real-valued coordinate - the policy SHAC differentiates through.

    Reads raw coordinates rather than the canvas, and that is not a style choice here: `render` is
    a comparison, so a canvas-reading policy has no gradient path from the placement it sees back
    to the actions that made it, and SHAC would lose backpropagation through time while looking
    identical from outside. See placax_agents/policy/architectures/continuous.py.
    """
    from placax_agents.policy.architectures.continuous import ContinuousActorCritic

    return ContinuousActorCritic(
        grid_x=benchmark.params.grid_x, grid_y=benchmark.params.effective_grid_y,
        cell_size=benchmark.cell_size, size_scale=float(benchmark.sizes_array.max()),
        features=features, num_layers=num_layers, init_log_std=init_log_std,
    )


POLICIES = {
    "cnn": _policy_cnn,
    "continuous": _policy_continuous,
    "oriented_cnn": _policy_oriented_cnn,
    "mlp": _policy_mlp,
    "wiremask_cnn": _policy_wiremask_cnn,
    "resnet_coarse_fine": _policy_resnet_coarse_fine,
}

# ---------------------------------------------------------------------------
# Optimizer.  (**kwargs) -> optax.GradientTransformation
# ---------------------------------------------------------------------------

MASKPLACE_LEARNING_RATE = 2.5e-3
"""MaskPlace's own --lr default (PPO2.py)."""

MASKPLACE_MAX_GRAD_NORM = 0.5
"""MaskPlace's own PPO.max_grad_norm; clipped per-network, not jointly."""


def _optimizer_adam(learning_rate: float = 3e-4, **_unused_context):
    return optax.adam(learning_rate)


def _optimizer_maskplace_split(
    learning_rate: float = MASKPLACE_LEARNING_RATE,
    max_grad_norm: float = MASKPLACE_MAX_GRAD_NORM,
    critic_param_prefix: str = "critic_",
    value_coef: float = 0.5,
):
    """Separately-clipped Adam for actor and critic, matching MaskPlace's two-independent-
    backward-pass setup.

    The critic's threshold is scaled by value_coef because ppo_loss hands this optimizer the
    value gradient already multiplied by value_coef, while PPO2.py clips the *unscaled* one at
    max_grad_norm: clipping value_coef*g_v at value_coef*max_grad_norm triggers on exactly the
    same ||g_v|| > 0.5 that PPO2.py's clip_grad_norm_(critic_net.parameters(), 0.5) does. The
    surviving value_coef factor on the update itself is then absorbed by Adam, which is
    invariant to a constant gradient rescale.
    """
    actor_chain = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate))
    critic_chain = optax.chain(
        optax.clip_by_global_norm(max_grad_norm * value_coef), optax.adam(learning_rate)
    )
    return make_grouped_optimizer(critic_chain, actor_chain, critic_param_prefix)


OPTIMIZERS = {"adam": _optimizer_adam, "maskplace_split": _optimizer_maskplace_split}

# ---------------------------------------------------------------------------
# Physical flow.  (**kwargs) -> CellPlacer / Validator
# ---------------------------------------------------------------------------
#
# Registered for the same reason as everything else here: a PPA number is only attributable if
# the tools that produced it are named in the run's own config, rather than chosen by whichever
# CLI flags a downstream script happened to be given. Imported lazily so that neither DREAMPlace
# nor OpenROAD is needed to import this module or to run a proxy-only experiment.
#
# A Spec's kwargs carry only what CHANGES THE RESULT and therefore belongs in the environment
# hash - target_density, the liberty file, the clock period. Where the binary lives on this
# particular machine (dreamplace_root, openroad_binary, use_docker, gpu) does not: two labs
# running the same experiment on the same design should compare as comparable, and a wheel path
# in an experiment hash would make that impossible. Those arrive separately, as `machine` kwargs,
# and are recorded in the run's fingerprint rather than its config.


def _cell_placer_dreamplace(dreamplace_root=None, target_density: float = 1.0, **kwargs):
    """DREAMPlace, with the one setting that changes the result named in the signature.

    `target_density` is spelled out rather than left to **kwargs so that `defaults.py` can see it:
    a Spec's unstated kwargs are completed from its builder's own signature before hashing, which
    is how `Spec("dreamplace")` and `Spec("dreamplace", {"target_density": 1.0})` come out as the
    one setup they are. Everything still arriving through **kwargs is this machine's - where
    DREAMPlace is installed, whether to run it in Docker or on the GPU - and is deliberately
    absent from the hash. See defaults.MACHINE_PARAMS.
    """
    from placax_tools.dreamplace.cell_placer import DREAMPlaceCellPlacer

    if dreamplace_root is None:
        raise ValueError(
            "the dreamplace cell placer needs dreamplace_root (or use_docker), which is a "
            "property of THIS MACHINE, not of the experiment - pass it as a machine kwarg to "
            "build_physical()/evaluate_physical(), e.g. from scripts/validate_design.py's "
            "--dreamplace_root or --use_docker."
        )
    return DREAMPlaceCellPlacer(
        dreamplace_root=dreamplace_root, target_density=target_density, **kwargs
    )


def _validator_openroad(
    liberty_path: str | None = None,
    clock_period_ns: float | None = None,
    wire_rc_layer: str = "metal3",
    clock_name: str = "core_clock",
    route: str | None = None,
    **kwargs,
):
    """OpenROAD, with everything that changes the measurement named here and therefore hashed.

    Timing needs `liberty_path` and `clock_period_ns`; `route` decides whether routed wirelength
    and DRC are measured at all. All four are properties of the EXPERIMENT - two PPA numbers taken
    at different routing depths are not the same measurement - so they belong in the Spec and in
    the hash, while `openroad_binary` arrives through **kwargs as a property of this host.
    """
    from placax_tools.openroad.validator import OpenROADValidator

    return OpenROADValidator(
        liberty_path=liberty_path, clock_period_ns=clock_period_ns,
        wire_rc_layer=wire_rc_layer, clock_name=clock_name, route=route, **kwargs
    )


CELL_PLACERS = {"dreamplace": _cell_placer_dreamplace}
VALIDATORS = {"openroad": _validator_openroad}

# ---------------------------------------------------------------------------
# Value loss, named so PPOConfig survives a JSON round trip.
# ---------------------------------------------------------------------------

VALUE_LOSSES = {"mse": mse_value_loss, "huber": huber_value_loss}


SLOTS = {
    "order": "ORDERS", "reward": "REWARDS", "state": "STATES", "action_mask": "MASKS",
    "initial_placement": "INITS", "action_space": "ACTION_SPACES", "legalization": "LEGALIZERS",
    "cell_placer": "CELL_PLACERS", "validator": "VALIDATORS", "policy": "POLICIES",
    "optimizer": "OPTIMIZERS", "value_loss": "VALUE_LOSSES",
    # These two live in build.py, next to the machinery they construct.
    "algorithm": "AGENTS", "loop": "LOOPS",
}
"""Config field -> the registry that field's names are looked up in. What `register` dispatches on."""


def _registry_for(slot: str) -> dict:
    if slot not in SLOTS:
        raise KeyError(f"unknown slot {slot!r}; choose one of {', '.join(sorted(SLOTS))}")
    name = SLOTS[slot]
    if name in ("AGENTS", "LOOPS"):
        # By module path, not `from ... import build`: the package's __init__ re-exports `build`
        # the FUNCTION, which would shadow the module and make this an AttributeError.
        import importlib

        return getattr(importlib.import_module("placax_agents.experiment.build"), name)
    return globals()[name]


def register(slot: str, name: str, builder) -> None:
    """Add a component of your own, from your own code, without editing this file.

    The registries are open, and this is the supported way in. A research script defines its
    reward (or state, policy, agent, action space...) as an ordinary function and registers it
    under a name; from then on a config can select it exactly like a shipped one, and it round
    trips through JSON like a shipped one, because a config holds names and not objects.

        from placax_agents.experiment.registry import register

        def my_reward(grid, weight: float = 2.0):
            return functools.partial(make_scaled_hpwl_reward, dense=True, reward_scale=weight)

        register("reward", "my_reward", my_reward)
        config = presets.training(..., )  # then Spec("my_reward", {"weight": 3.0})

    Use this rather than mutating the dict directly: `defaults_for` memoizes what it finds, so a
    name that was hashed before it existed would keep hashing against an empty default set. This
    clears that cache; a bare `REWARDS[name] = fn` does not.

    **The reproducibility caveat, stated plainly.** A config records the NAME. Two runs whose
    configs both say `my_reward` compare as comparable, and whether they really used the same
    function depends on the code that registered it - which the run's manifest pins only as far
    as `fingerprint()["git_revision"]` reaches, and a script outside this repository is not in
    that. Shipped components do not have this problem. If a custom component matters to a result,
    version it with the same care as the result.
    """
    registry = _registry_for(slot)
    if not callable(builder):
        raise TypeError(f"a {slot} builder must be callable, got {type(builder).__name__}")
    registry[name] = builder
    # Defaults are memoized per (slot, name); a stale empty entry would silently change the hash.
    from placax_agents.experiment.defaults import defaults_for

    defaults_for.cache_clear()


def registered(slot: str) -> list[str]:
    """Every name currently available for `slot`, shipped and registered alike."""
    return sorted(_registry_for(slot))


def resolve(registry: dict, spec, *args, what: str = "component"):
    """Looks up spec.name in registry and calls it with spec.kwargs, with a readable error."""
    if spec.name not in registry:
        raise KeyError(
            f"unknown {what} {spec.name!r}; registered: {', '.join(sorted(registry))}. "
            f"Register your own from your own code with "
            f"placax_agents.experiment.registry.register('<slot>', {spec.name!r}, builder) - "
            f"the registries are open, and nothing here needs editing to add one."
        )
    return registry[spec.name](*args, **spec.kwargs)
