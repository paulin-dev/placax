# Scratch utility: allocate a JAX array and hold it so GPU memory use is
# visible in nvidia-smi.
import placax  # runs the device setup in _device.py first
import jax.numpy as jnp

x = jnp.ones((30000, 20))
x.block_until_ready()  # JAX is async - force the allocation now

print(f"Allocated on: {x.device}")
print(f"Array size: {x.nbytes / 1e6:.1f} MB")
print("Check `nvidia-smi` in another terminal now.")
input("Press Enter here to release the array and exit...")
