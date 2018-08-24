import tensorflow as tf


def on_each_gpu(func):
  """A Decorator which perform func independently on each gpu.

  This function will call func on each gpu, which is useful for
  multi-gpu computations. The decorator takes a list of tensor 
  as inputs. See examples in network/multigpu/layers.py.

  Args:
    func: A tensorflow operation which is performed on a single GPU, and
      the function takes a tensor as input.
  """
  def inner(*args, **kwargs):
    xs = args[0]
    assert(isinstance(xs, list))
    outputs = []
    for x in xs:
      with tf.device(x.device):
        outputs.append(func(x, *args[1:], **kwargs))

    return outputs

  return inner
