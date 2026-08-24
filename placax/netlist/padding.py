"""Bridges load_netlist()'s named/dict-based output into the padded,
index-based arrays hpwl(), wiremask(), and render() consume. One
canonical macro_name -> index mapping is used throughout, so a macro's
row in positions/sizes_array always matches its index in padded_pin_idx."""
from placax import _device  # noqa: F401  must run before any `import jax` below

import jax
import jax.numpy as jnp
import numpy as np

from placax.types import Nets, SizeMap


def _build_sizes_array(macro_sizes: SizeMap, macro_names: list[str]) -> np.ndarray:
    return np.array([macro_sizes[name] for name in macro_names], dtype=np.float32)


def build_padded_arrays(macro_sizes: SizeMap, nets: Nets):
    """Returns (name_to_idx, sizes_array, padded_pin_idx, padded_pin_offset,
    valid_mask). Built with NumPy, not incremental JAX .at[].set() calls -
    each JAX update allocates a whole new array, far too slow at real
    benchmark scale (thousands of pins); converted to JAX arrays once,
    at the end."""
    macro_names = sorted(macro_sizes)
    name_to_idx = {name: i for i, name in enumerate(macro_names)}
    sizes_array = _build_sizes_array(macro_sizes, macro_names)

    n_nets = len(nets)
    max_pins = max((len(net) for net in nets), default=0)

    padded_pin_idx = np.zeros((n_nets, max_pins), dtype=np.int32)
    padded_pin_offset = np.zeros((n_nets, max_pins, 2), dtype=np.float32)
    valid_mask = np.zeros((n_nets, max_pins), dtype=bool)

    for i, net in enumerate(nets):
        for j, (name, x_off, y_off) in enumerate(net):
            padded_pin_idx[i, j] = name_to_idx[name]
            padded_pin_offset[i, j] = (x_off, y_off)
            valid_mask[i, j] = True

    return (
        name_to_idx,
        jnp.array(sizes_array),
        jnp.array(padded_pin_idx),
        jnp.array(padded_pin_offset),
        jnp.array(valid_mask),
    )


def _group_pins_by_macro(
    padded_pin_idx: jax.Array, padded_pin_offset: jax.Array, valid_mask: jax.Array, n_macros: int
) -> list[list[tuple[int, float, float]]]:
    """Returns, per macro, [(net_idx, x_offset, y_offset)] - only the
    real (non-padding) entries. Flattened first: a Python loop over the
    full padded grid would be far slower than one over just the real
    entries."""
    n_nets = padded_pin_idx.shape[0]
    pin_idx_np, offset_np, valid_np = map(np.array, (padded_pin_idx, padded_pin_offset, valid_mask))

    net_idx_grid = np.broadcast_to(np.arange(n_nets)[:, None], pin_idx_np.shape)
    valid_flat = valid_np.ravel()
    net_ids = net_idx_grid.ravel()[valid_flat]
    macro_ids = pin_idx_np.ravel()[valid_flat]
    offsets = offset_np.reshape(-1, 2)[valid_flat]

    participations: list[list[tuple[int, float, float]]] = [[] for _ in range(n_macros)]
    for net_idx, macro_idx, x_off, y_off in zip(
        net_ids.tolist(), macro_ids.tolist(), offsets[:, 0].tolist(), offsets[:, 1].tolist()
    ):
        participations[macro_idx].append((net_idx, x_off, y_off))
    return participations


def build_macro_net_index(
    padded_pin_idx: jax.Array, padded_pin_offset: jax.Array, valid_mask: jax.Array, n_macros: int
):
    """For each macro, which nets (and at what offset) it participates in -
    padded to max_participation, the most nets any single macro touches
    (much smaller than max_pins, the netlist's own worst case). Lets
    wiremask() touch only one macro's small participation list per
    candidate cell instead of the whole netlist."""
    participations = _group_pins_by_macro(padded_pin_idx, padded_pin_offset, valid_mask, n_macros)

    max_participation = max((len(p) for p in participations), default=0)
    macro_net_idx = np.zeros((n_macros, max_participation), dtype=np.int32)
    macro_net_offset = np.zeros((n_macros, max_participation, 2), dtype=np.float32)
    macro_net_valid = np.zeros((n_macros, max_participation), dtype=bool)

    for macro_idx, parts in enumerate(participations):
        for slot, (net_idx, x_off, y_off) in enumerate(parts):
            macro_net_idx[macro_idx, slot] = net_idx
            macro_net_offset[macro_idx, slot] = (x_off, y_off)
            macro_net_valid[macro_idx, slot] = True

    return jnp.array(macro_net_idx), jnp.array(macro_net_offset), jnp.array(macro_net_valid)
