"""Bridges load_netlist()'s named/dict output into the padded, index-based arrays hpwl()/wiremask()/render() consume."""
from placax import _device  # noqa: F401  must run before any `import jax` below

import jax
import jax.numpy as jnp
import numpy as np

from placax.netlist.order import alphabetical_order
from placax.types import Nets, OrderFn, SizeMap


def _pack_groups(
    groups: list[list[tuple[int, float, float]]], n_groups: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pads per-group (index, x_offset, y_offset) entries into aligned (idx, offset, valid) arrays."""
    # Every group is padded to max_len with zeros and marked invalid, forming a rectangular array.
    max_len = max((len(g) for g in groups), default=0)
    idx = np.zeros((n_groups, max_len), dtype=np.int32)
    offset = np.zeros((n_groups, max_len, 2), dtype=np.float32)
    valid = np.zeros((n_groups, max_len), dtype=bool)

    # valid stays False past each group's own length - that's the padding.
    for i, group in enumerate(groups):
        for j, (member_idx, x_off, y_off) in enumerate(group):
            idx[i, j] = member_idx
            offset[i, j] = (x_off, y_off)
            valid[i, j] = True
    return idx, offset, valid


def _order_macros(macro_sizes: SizeMap, nets: Nets, order_fn: OrderFn) -> dict[str, int]:
    """Returns name -> row index per order_fn; that same row index is reused in sizes_array/positions."""
    macro_names = order_fn(macro_sizes, nets)
    assert set(macro_names) == set(macro_sizes), "order_fn must return every macro exactly once"
    return {name: i for i, name in enumerate(macro_names)}


def _remap_nets_to_indices(nets: Nets, name_to_idx: dict[str, int]) -> list[list[tuple[int, float, float]]]:
    """Remaps each net's macro names to the row indices sizes_array/positions use."""
    return [[(name_to_idx[name], x_off, y_off) for name, x_off, y_off in net] for net in nets]


def build_padded_arrays(macro_sizes: SizeMap, nets: Nets, order_fn: OrderFn = alphabetical_order):
    """Returns (name_to_idx, sizes_array, padded_pin_idx, padded_pin_offset, valid_mask) for hpwl()/wiremask()/render()."""
    # 1. Decide the macro-to-row-index mapping used by every array below.
    name_to_idx = _order_macros(macro_sizes, nets, order_fn)
    sizes_array = jnp.array(
        np.array([macro_sizes[name] for name in name_to_idx], dtype=np.float32)
    )
    # 2. Remap nets from macro names to row indices, then pad them into rectangular arrays.
    net_groups = _remap_nets_to_indices(nets, name_to_idx)
    pin_idx, pin_offset, valid_mask = _pack_groups(net_groups, len(nets))

    # 3. Everything is built with NumPy and converted to JAX arrays only once here.
    return name_to_idx, sizes_array, jnp.array(pin_idx), jnp.array(pin_offset), jnp.array(valid_mask)


def build_macro_net_index(
    padded_pin_idx: jax.Array, padded_pin_offset: jax.Array, valid_mask: jax.Array, n_macros: int
):
    """Builds the per-macro reverse index of which nets each macro participates in, and at what offset."""
    # 1. Drop back to just the real (valid) pin entries first - far fewer than the full padded grid.
    pin_idx_np, offset_np, valid_np = map(np.array, (padded_pin_idx, padded_pin_offset, valid_mask))
    n_nets = pin_idx_np.shape[0]

    valid_flat = valid_np.ravel()
    net_ids = np.broadcast_to(np.arange(n_nets)[:, None], pin_idx_np.shape).ravel()[valid_flat]
    macro_ids = pin_idx_np.ravel()[valid_flat]
    offsets = offset_np.reshape(-1, 2)[valid_flat]

    # 2. Invert the net->macro relationship: for each macro, collect its (net, offset) pairs.
    participations: list[list[tuple[int, float, float]]] = [[] for _ in range(n_macros)]
    for net_idx, macro_idx, x_off, y_off in zip(
        net_ids.tolist(), macro_ids.tolist(), offsets[:, 0].tolist(), offsets[:, 1].tolist()
    ):
        participations[macro_idx].append((net_idx, x_off, y_off))

    # 3. Pad to a rectangular array again, sized to the busiest macro.
    macro_net_idx, macro_net_offset, macro_net_valid = _pack_groups(participations, n_macros)
    return jnp.array(macro_net_idx), jnp.array(macro_net_offset), jnp.array(macro_net_valid)
