import tensorflow as tf

from network.common.resnet_v1 import resnet_v1_101
import network.common.layers as nn

def _fcn_builder(x,
                 name,
                 cnn_fn,
                 num_classes,
                 is_training,
                 use_global_status,
                 reuse=False):
  """Helper function to build FCN8s model for semantic segmentation.

  The FCN8s model is composed of one base network (ResNet101) and 
  one classifier.

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
  h, w = x.get_shape().as_list()[1:3] # NxHxWxC
  assert(h%48 == 0 and w%48 == 0 and h == w)

  # Build the base network.
  x = cnn_fn(x, name, is_training, use_global_status, reuse)

  with tf.variable_scope(name, reuse=reuse) as scope:
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


def fcn8s_resnet101(x,
                    num_classes,
                    is_training,
                    use_global_status,
                    reuse=False):
  """Builds FCN8s model based on ResNet101.

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
  scores = []
  with tf.name_scope('scale_0') as scope:
    score = _fcn_builder(
        x,
        'resnet_v1_101',
        resnet_v1_101,
        num_classes,
        is_training,
        use_global_status,
        reuse=reuse)

    scores.append(score)

  return scores
