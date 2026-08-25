"""Per-parameter-group optimizers - e.g. a critic optimized genuinely
independently of the actor, without a second backward pass. General on
purpose: works for any architecture that names its own parameter groups
(a naming convention, not a base class or special API), any grouping
scheme, and any pair of optax transforms - not specific to actor/critic,
and not specific to any one policy module."""
from collections.abc import Hashable

import jax
import optax


def label_params_by_name_prefix(
    params, prefix: str, matched_label: Hashable = "matched", default_label: Hashable = "default"
):
    """Labels every leaf by whether any key in its path starts with
    prefix - e.g. prefix="critic_" matches a leaf under
    variables["params"]["critic_value"]["kernel"] regardless of how
    deeply that sits inside a larger variables tree (params/batch_stats
    wrapping, an outer module, ...). Returns a label pytree matching
    params' structure, suitable for optax.multi_transform's param_labels."""

    def label_leaf(path, _leaf):
        keys = (str(getattr(entry, "key", entry)) for entry in path)
        return matched_label if any(key.startswith(prefix) for key in keys) else default_label

    return jax.tree_util.tree_map_with_path(label_leaf, params)


def make_grouped_optimizer(
    matched_transform: optax.GradientTransformation,
    default_transform: optax.GradientTransformation,
    prefix: str,
) -> optax.GradientTransformation:
    """One optax.GradientTransformation that applies matched_transform to
    every parameter named under prefix (see label_params_by_name_prefix)
    and default_transform to everything else - each with its own
    optimizer state (e.g. its own Adam moments), each seeing only its own
    group's gradients.

    This is mathematically identical to computing and applying each
    group's update via a fully separate backward pass, not an
    approximation of it: a loss term that doesn't depend on a parameter
    has exactly zero gradient w.r.t. it (basic chain rule), so as long as
    the two groups don't share parameters, one joint jax.grad() call
    already yields exactly the per-group gradients two separate
    .backward() calls would - what differs is only what happens to each
    group's gradient afterward, which is exactly what this function lets
    you control per group."""
    return optax.multi_transform(
        {"matched": matched_transform, "default": default_transform},
        lambda params: label_params_by_name_prefix(params, prefix, "matched", "default"),
    )
