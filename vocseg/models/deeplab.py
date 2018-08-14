import tensorflow as tf

from network.common.resnet_v1 import resnet_v1_101
import network.common.layers as nn

def _deeplab_builder(x,
                     name,
                     cnn_fn,
                     num_classes,
                     is_training,
                     use_global_status,
                     reuse=False):
  """Helper function to build Deeplab v2 model for semantic segmentation.

  The Deeplab v2 model is composed of one base network (ResNet101) and 
  one ASPP module (4 Atrous Convolutional layers of different size). The
  segmentation prediction is the summation of 4 outputs of the ASPP module.

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
  # Build the base network.
  x = cnn_fn(x, name, is_training, use_global_status, reuse)
  
  with tf.variable_scope(name, reuse=reuse) as scope:
    # Build the ASPP module.
    aspp = []
    for i,dilation in enumerate([6, 12, 18, 24]):
      score = nn.atrous_conv(
          x,
          name='fc1_c{:d}'.format(i),
          filters=num_classes,
          kernel_size=3,
          dilation=dilation,
          padding='SAME',
          relu=False,
          biased=True,
          bn=False,
          is_training=is_training)
      aspp.append(score)

    score = tf.add_n(aspp, name='fc1_sum')

  return score


def deeplab_resnet101(x,
                      num_classes,
                      is_training,
                      use_global_status,
                      reuse=False):
  """Builds Deeplab v2 based on ResNet101.

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels].
    name: The prefix of tensorflow variables defined in this network.
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
  h, w = x.get_shape().as_list()[1:3] # NxHxWxC

  scores = []
  for i,scale in enumerate([1]):
    with tf.name_scope('scale_{:d}'.format(i)) as scope:
      x_in = x

      score = _deeplab_builder(
          x_in,
          'resnet_v1_101',
          resnet_v1_101,
          num_classes,
          is_training,
          use_global_status,
          reuse=reuse)

      scores.append(score)

  return scores
