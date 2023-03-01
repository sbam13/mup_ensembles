import jax
import jax.numpy as jnp
import collections
import itertools
# import jax.lib.xla_bridge as xb

# def _pmap_device_order():
#   # match the default device assignments used in pmap:
#   # for single-host, that's the XLA default device assignment
#   # for multi-host, it's the order of jax.local_devices()
#   if jax.process_count() == 1:
#     return [d for d in xb.get_backend().get_default_device_assignment(
#         jax.device_count()) if d.process_index == jax.process_index()]
#   else:
#     return jax.local_devices()

def prefetch_to_device(iterator, size):
  """Shard and prefetch batches on device.

  This utility takes an iterator and returns a new iterator which fills an on
  device prefetch buffer. Eager prefetching can improve the performance of
  training loops significantly by overlapping compute and data transfer.

  This utility is mostly useful for GPUs, for TPUs and CPUs it should not be
  necessary -- the TPU & CPU memory allocators (normally) don't pick a memory
  location that isn't free yet so they don't block. Instead those allocators OOM.

  Args:
    iterator: an iterator that yields a pytree of ndarrays where the first
      dimension is sharded across devices.

    size: the size of the prefetch buffer.

      If you're training on GPUs, 2 is generally the best choice because this
      guarantees that you can overlap a training step on GPU with a data
      prefetch step on CPU.

    devices: the list of devices to which the arrays should be prefetched.

      Defaults to the order of devices expected by `jax.pmap`.

  Yields:
    The original items from the iterator where each ndarray is now a sharded to
    the specified devices.
  """
  queue = collections.deque()
  # devices = devices or _pmap_device_order()

#   def _prefetch(x_ch):
#     x_jnp = jnp.array(x_ch)
#     return jax.device_put_replicated(x_jnp, devices)

  def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
    for data in itertools.islice(iterator, n):
      x_ch, y_list = data
      x_jnp = jnp.array(x_ch)
      y_jnp = jnp.array(y_list)
      queue.append(jax.device_put((x_jnp, y_jnp)))

  enqueue(size)  # Fill up the buffer.
  while queue:
    yield queue.popleft()
    enqueue(1)