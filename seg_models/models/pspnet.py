import tensorflow as tf

from network.common.resnet_v1 import resnet_v1_101
import network.common.layers as nn

def _pspnet_builder(x,
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
  h, w = x.get_shape().as_list()[1:3] # NxHxWxC
  assert(h%48 == 0 and w%48 == 0 and h == w)

  # Build the base network.
  x = cnn_fn(x, name, is_training, use_global_status, reuse)

  with tf.variable_scope(name, reuse=reuse) as scope:
    # Build the PSP module
    pool_k = int(h/8) # the base network is stride 8 by default.

    # Build pooling layer results in 1x1 output.
    pool1 = tf.nn.avg_pool(x,
                           name='block5/pool1',
                           ksize=[1,pool_k,pool_k,1],
                           strides=[1,pool_k,pool_k,1],
                           padding='VALID')
    pool1 = nn.conv(pool1,
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
    pool1 = tf.image.resize_bilinear(pool1, [pool_k, pool_k])

    # Build pooling layer results in 2x2 output.
    pool2 = tf.nn.avg_pool(x,
                           name='block5/pool2',
                           ksize=[1,pool_k//2,pool_k//2,1],
                           strides=[1,pool_k//2,pool_k//2,1],
                           padding='VALID')
    pool2 = nn.conv(pool2,
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
    pool2 = tf.image.resize_bilinear(pool2, [pool_k, pool_k])

    # Build pooling layer results in 3x3 output.
    pool3 = tf.nn.avg_pool(x,
                           name='block5/pool3',
                           ksize=[1,pool_k//3,pool_k//3,1],
                           strides=[1,pool_k//3,pool_k//3,1],
                           padding='VALID')
    pool3 = nn.conv(pool3,
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
    pool3 = tf.image.resize_bilinear(pool3, [pool_k, pool_k])

    # Build pooling layer results in 6x6 output.
    pool6 = tf.nn.avg_pool(x,
                           name='block5/pool6',
                           ksize=[1,pool_k//6,pool_k//6,1],
                           strides=[1,pool_k//6,pool_k//6,1],
                           padding='VALID')
    pool6 = nn.conv(pool6,
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
    pool6 = tf.image.resize_bilinear(pool6, [pool_k, pool_k])

    # Fuse the pooled feature maps with its input, and generate
    # segmentation prediction.
    x = tf.concat([pool1, pool2, pool3, pool6, x],
                  name='block5/concat',
                  axis=3)
    x = nn.conv(x,
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
    x = nn.conv(x,
                'block5/fc1_voc12',
                num_classes,
                1,
                1,
                padding='SAME',
                biased=True,
                bn=False,
                relu=False,
                is_training=is_training)

    return x


def pspnet_resnet101(x,
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

  scores = []
  with tf.name_scope('scale_0') as scope:
    score = _pspnet_builder(
        x,
        'resnet_v1_101',
        resnet_v1_101,
        num_classes,
        is_training,
        use_global_status,
        reuse=reuse)

    scores.append(score)

  return scores
