import tensorflow as tf

from network.multigpu.resnet_v1 import resnet_v1_101
import network.multigpu.layers as nn_mgpu
from network.multigpu.utils import on_each_gpu


@on_each_gpu
def avg_pools(x,
              name,
              kernel_size,
              strides,
              padding):
  k = kernel_size
  s = strides
  return tf.nn.avg_pool(x,
                        name=name,
                        ksize=[1,k,k,1],
                        strides=[1,s,s,1],
                        padding=padding)


@on_each_gpu
def upsample_bilinears(x,
                       new_h,
                       new_w):
  return tf.image.resize_bilinear(x, [new_h, new_w])


def _pspnet_builder(xs,
                    name,
                    cnn_fn,
                    num_classes,
                    is_training,
                    use_global_status,
                    reuse=False):
  """Helper function to build PSPNet model for semantic segmentation.

  The PSPNet model is composed of one base network (ResNet101) and 
  one pyramid spatial pooling (PSP) module, followed with concatenation
  and two more convlutional layers for segmentation prediction.

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels].
    name: The prefix of tensorflow variables defined in this network.
    cnn_fn: A function which builds the base network (ResNet101).
    num_classes: Number of predicted classes for classification tasks.
    is_training: If the tensorflow variables defined in this network
      would be used for training.
    use_global_status: enable/disable use_global_status for batch
      normalization. If True, moving mean and moving variance are updated
      by exponential decay.
    reuse: enable/disable reuse for reusing tensorflow variables. It is 
      useful for sharing weight parameters across two identical networks.

  Returns:
    A tensor of size [batch_size, height_in/8, width_in/8, num_classes].
  """
  # Ensure that the size of input data is valid (should be multiple of 6x8=48).
  h, w = xs[0].get_shape().as_list()[1:3] # NxHxWxC
  assert(h%48 == 0 and w%48 == 0 and h == w)

  # Build the base network.
  xs = cnn_fn(xs, name, is_training, use_global_status, reuse)

  with tf.variable_scope(name, reuse=reuse) as scope:
    # Build the PSP module
    pool_k = int(h/8) # the base network is stride 8 by default.

    # Build pooling layer results in 1x1 output.
    pool1s = avg_pools(xs,
                       'block5/pool1',
                       pool_k,
                       pool_k,
                       'VALID')
    pool1s = nn_mgpu.conv(pool1s,
                          'block5/pool1/conv1',
                          512,
                          1,
                          1,
                          padding='SAME',
                          biased=False,
                          bn=True,
                          relu=True,
                          is_training=is_training,
                          decay=0.99,
                          use_global_status=use_global_status)
    pool1s = upsample_bilinears(pool1s, pool_k, pool_k)

    # Build pooling layer results in 2x2 output.
    pool2s = avg_pools(xs,
                       'block5/pool2',
                       pool_k//2,
                       pool_k//2,
                       'VALID')
    pool2s = nn_mgpu.conv(pool2s,
                          'block5/pool2/conv1',
                          512,
                          1,
                          1,
                          padding='SAME',
                          biased=False,
                          bn=True,
                          relu=True,
                          is_training=is_training,
                          decay=0.99,
                          use_global_status=use_global_status)
    pool2s = upsample_bilinears(pool2s, pool_k, pool_k)

    # Build pooling layer results in 3x3 output.
    pool3s = avg_pools(xs,
                       'block5/pool3',
                       pool_k//3,
                       pool_k//3,
                       'VALID')
    pool3s = nn_mgpu.conv(pool3s,
                          'block5/pool3/conv1',
                          512,
                          1,
                          1,
                          padding='SAME',
                          biased=False,
                          bn=True,
                          relu=True,
                          is_training=is_training,
                          decay=0.99,
                          use_global_status=use_global_status)
    pool3s = upsample_bilinears(pool3s, pool_k, pool_k)

    # Build pooling layer results in 6x6 output.
    pool6s = avg_pools(xs,
                       'block5/pool6',
                       pool_k//6,
                       pool_k//6,
                       'VALID')
    pool6s = nn_mgpu.conv(pool6s,
                          'block5/pool6/conv1',
                          512,
                          1,
                          1,
                          padding='SAME',
                          biased=False,
                          bn=True,
                          relu=True,
                          is_training=is_training,
                          decay=0.99,
                          use_global_status=use_global_status)
    pool6s = upsample_bilinears(pool6s, pool_k, pool_k)

    # Fuse the pooled feature maps with its input, and generate
    # segmentation prediction.
    xs = nn_mgpu.concat(
        [pool1s, pool2s, pool3s, pool6s, xs],
        name='block5/concat',
        axis=3)
    xs = nn_mgpu.conv(xs,
                      'block5/conv2',
                      512,
                      3,
                      1,
                      padding='SAME',
                      biased=False,
                      bn=True,
                      relu=True,
                      is_training=is_training,
                      decay=0.99,
                      use_global_status=use_global_status)
    xs = nn_mgpu.conv(xs,
                      'block5/fc1_voc12',
                      num_classes,
                      1,
                      1,
                      padding='SAME',
                      biased=True,
                      bn=False,
                      relu=False,
                      is_training=is_training)

    return xs


def pspnet_resnet101(xs,
                     num_classes,
                     is_training,
                     use_global_status,
                     reuse=False):
  """Helper function to build PSPNet model for semantic segmentation.

  The PSPNet model is composed of one base network (ResNet101) and 
  one pyramid spatial pooling (PSP) module, followed with concatenation
  and two more convlutional layers for segmentation prediction.

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels].
    num_classes: Number of predicted classes for classification tasks.
    is_training: If the tensorflow variables defined in this network
      would be used for training.
    use_global_status: enable/disable use_global_status for batch
      normalization. If True, moving mean and moving variance are updated
      by exponential decay.
    reuse: enable/disable reuse for reusing tensorflow variables. It is 
      useful for sharing weight parameters across two identical networks.

  Returns:
    A tensor of size [batch_size, height_in/8, width_in/8, num_classes].
  """

  num_gpu = len(xs)
  scores = []
  with tf.name_scope('scale_0') as scope:
    score = _pspnet_builder(
        xs,
        'resnet_v1_101',
        resnet_v1_101,
        num_classes,
        is_training,
        use_global_status,
        reuse=reuse)
    for s in score:
      scores.append([s])

  return scores
