"""Per-parameter-group optimizers, e.g. a separately-optimized critic
without a second backward pass."""
from collections.abc import Hashable

import jax
import optax


def label_params_by_name_prefix(
    params, prefix: str, matched_label: Hashable = "matched", default_label: Hashable = "default"
):
    """Labels every leaf matched_label if any key in its path starts with
    prefix, else default_label. Returns a label pytree matching params'
    structure, suitable for optax.multi_transform's param_labels."""

    def label_leaf(path, _leaf):
        keys = (str(getattr(entry, "key", entry)) for entry in path)
        return matched_label if any(key.startswith(prefix) for key in keys) else default_label

    return jax.tree_util.tree_map_with_path(label_leaf, params)


def make_grouped_optimizer(
    matched_transform: optax.GradientTransformation,
    default_transform: optax.GradientTransformation,
    prefix: str,
) -> optax.GradientTransformation:
    """Builds one optimizer applying matched_transform to params under
    prefix and default_transform to the rest, each with independent
    optimizer state; equivalent to a separate backward pass per group
    since unrelated params have zero gradient."""
    return optax.multi_transform(
        {"matched": matched_transform, "default": default_transform},
        lambda params: label_params_by_name_prefix(params, prefix, "matched", "default"),
    )
