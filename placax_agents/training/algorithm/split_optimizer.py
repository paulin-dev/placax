"""Per-parameter-group optimizers, e.g. a separately-optimized critic
without a second backward pass."""
from collections.abc import Hashable

import jax
import optax


def label_params_by_name_prefix(
    params, prefix: str, matched_label: Hashable = "matched", default_label: Hashable = "default"
):
    """Labels each param leaf matched_label if any key in its path starts with prefix, else default_label."""

    def label_leaf(path, _leaf):
        # path is a sequence of pytree keys (e.g. dict keys) leading to this leaf;
        # stringify each one so we can check it against the prefix.
        keys = (str(getattr(entry, "key", entry)) for entry in path)
        return matched_label if any(key.startswith(prefix) for key in keys) else default_label

    # Walk the whole params pytree, replacing each leaf with its label - the result
    # is a same-shaped pytree of labels, exactly what optax.multi_transform expects.
    return jax.tree_util.tree_map_with_path(label_leaf, params)


def make_grouped_optimizer(
    matched_transform: optax.GradientTransformation,
    default_transform: optax.GradientTransformation,
    prefix: str,
) -> optax.GradientTransformation:
    """Builds one optimizer applying matched_transform to params under prefix and default_transform to the rest."""
    return optax.multi_transform(
        {"matched": matched_transform, "default": default_transform},
        lambda params: label_params_by_name_prefix(params, prefix, "matched", "default"),
    )
