import math

import tensorflow as tf
import numpy as np


def batch_norm(x,
               name,
               activation_fn=None,
               decay=0.99,
               epsilon=0.001,
               is_training=True):
  """Batch normalization.

  This function perform batch normalization. If it is set for training,
  it will update moving mean and moving variance to keep track of global
  statistics by exponential decay.

  output =  [(x - mean) / sqrt(var)] * gamma + beta.

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels].
    name: The prefix of tensorflow variables defined in this layer.
    activation_fn: The non-linear function, such as tf.nn.relu. If 
      activation_fn is None, skip it and maintain a linear activation.
    decay: The exponential decay rate.
    epsilon: Small float added to variance to avoid dividing by zero.
    is_training: enable/disable is_training for updating moving mean and
      moving variance by exponential decay. If True, compute batch mean
      and batch variance per batch; otherwise, use moving mean and moving
      variance as batch mean and batch variance.

  Returns:
    A tensor of size [batch_size, height_in, width_in, channels]
  """
  with tf.variable_scope(name) as scope:
    shape_x = x.get_shape().as_list()

    beta = tf.get_variable(
        'beta',
        shape_x[-1],
        initializer=tf.constant_initializer(0.0),
        trainable=is_training)
    gamma = tf.get_variable(
        'gamma',
        shape_x[-1],
        initializer=tf.constant_initializer(1.0),
        trainable=is_training)
    moving_mean = tf.get_variable(
        'moving_mean',
        shape_x[-1],
        initializer=tf.constant_initializer(0.0),
        trainable=False)
    moving_var = tf.get_variable(
        'moving_variance',
        shape_x[-1],
        initializer=tf.constant_initializer(1.0),
        trainable=False)

    if is_training:
      # Update moving mean and variance before
      # applying batch normalization
      mean, var = tf.nn.moments(x,
                                np.arange(len(shape_x)-1),
                                keep_dims=True)
      mean = tf.reshape(mean,
                        [mean.shape.as_list()[-1]])
      var = tf.reshape(var,
                       [var.shape.as_list()[-1]])

      # Update moving mean and moving variance by exponential decay.
      update_moving_mean = tf.assign(
          moving_mean,
          moving_mean*decay + mean*(1-decay))
      update_moving_var = tf.assign(
          moving_var,
          moving_var*decay + var*(1-decay))
      update_ops = [update_moving_mean, update_moving_var]

      with tf.control_dependencies(update_ops):
        output = tf.nn.batch_normalization(x,
                                           mean,
                                           var,
                                           beta,
                                           gamma,
                                           epsilon)
    else:
      # Use collected moving mean and moving variance for normalization.
      mean = moving_mean
      var = moving_var

      output = tf.nn.batch_normalization(x,
                                         mean,
                                         var,
                                         beta,
                                         gamma,
                                         epsilon)

    # Apply activation_fn, if it is not None.
    if activation_fn:
      output = activation_fn(output)

  return output


def conv(x,
         name,
         filters,
         kernel_size,
         strides,
         padding,
         relu=True,
         biased=True,
         bn=True,
         decay=0.9997,
         is_training=True,
         use_global_status=True):
  """Convolutional layers with batch normalization and ReLU.

  This function perform convolution, batch_norm (if bn=True),
  and ReLU (if relu=True).

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels].
    name: The prefix of tensorflow variables defined in this layer.
    filters: A number indicating the number of output channels.
    kernel_size: A number indicating the size of convolutional kernels.
    strides: A number indicating the stride of the sliding window for
      height and width.
    padding: 'VALID' or 'SAME'.
    relu: enable/disable relu for ReLU as activation function. If relu 
      is False, maintain linear activation.
    biased: enable/disable biased for adding biases after convolution.
    bn: enable/disable bn for batch normalization.
    decay: A number indication decay rate for updating moving mean and 
      moving variance in batch normalization.
    is_training: If the tensorflow variables defined in this layer 
      would be used for training.
    use_global_status: enable/disable use_global_status for batch
      normalization. If True, moving mean and moving variance are updated
      by exponential decay.

  Returns:
    A tensor of size [batch_size, height_out, width_out, channels_out].
  """
  c_i = x.get_shape().as_list()[-1] # input channels
  c_o = filters # output channels

  # Define helper function.
  convolve = lambda i,k: tf.nn.conv2d(
      i,
      k,
      [1, strides, strides, 1],
      padding=padding)

  with tf.variable_scope(name) as scope:
    kernel = tf.get_variable(
        name='weights',
        shape=[kernel_size, kernel_size, c_i, c_o],
        trainable=is_training)

    if strides > 1:
      pad = kernel_size - 1
      pad_beg = pad // 2
      pad_end = pad - pad_beg
      pad_h = [pad_beg, pad_end]
      pad_w = [pad_beg, pad_end]
      x = tf.pad(x, [[0,0], pad_h, pad_w, [0,0]])

    output = convolve(x, kernel)

    # Add the biases.
    if biased:
      biases = tf.get_variable('biases', [c_o], trainable=is_training)
      output = tf.nn.bias_add(output, biases)

    # Apply batch normalization.
    if bn:
      is_bn_training = not use_global_status
      output = batch_norm(output,
                          'BatchNorm',
                          is_training=is_bn_training,
                          decay=decay,
                          activation_fn=None)

    # Apply ReLU as activation function.
    if relu:
      output = tf.nn.relu(output)

  return output


def atrous_conv(x,
                name,
                filters,
                kernel_size,
                dilation,
                padding,
                relu=True,
                biased=True,
                bn=True,
                decay=0.9997,
                is_training=True,
                use_global_status=True):
  """Atrous convolutional layers with batch normalization and ReLU.

  This function perform atrous convolution, batch_norm (if bn=True),
  and ReLU (if relu=True).

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels].
    name: The prefix of tensorflow variables defined in this layer.
    filters: A number indicating the number of output channels.
    kernel_size: A number indicating the size of convolutional kernels.
    dilation: A number indicating the dilation factor for height and width.
    padding: 'VALID' or 'SAME'.
    relu: enable/disable relu for ReLU as activation function. If relu 
      is False, maintain linear activation.
    biased: enable/disable biased for adding biases after convolution.
    bn: enable/disable bn for batch normalization.
    decay: A number indication decay rate for updating moving mean and 
      moving variance in batch normalization.
    is_training: If the tensorflow variables defined in this layer 
      would be used for training.
    use_global_status: enable/disable use_global_status for batch
      normalization. If True, moving mean and moving variance are updated
      by exponential decay.

  Returns:
    A tensor of size [batch_size, height_out, width_out, channels_out].
  """
  c_i = x.get_shape().as_list()[-1] # input channels
  c_o = filters # output channels

  # Define helper function.
  convolve = lambda i,k: tf.nn.atrous_conv2d(
      i,
      k,
      dilation,
      padding)

  with tf.variable_scope(name) as scope:
    kernel = tf.get_variable(
        name='weights',
        shape=[kernel_size, kernel_size, c_i, c_o],
        trainable=is_training,)
    output = convolve(x, kernel)

    # Add the biases.
    if biased:
      biases = tf.get_variable('biases', [c_o], trainable=is_training)
      output = tf.nn.bias_add(output, biases)

    # Apply batch normalization.
    if bn:
      is_bn_training = not use_global_status
      output = batch_norm(output, 'BatchNorm',
                          is_training=is_bn_training,
                          decay=decay,
                          activation_fn=None)

    # Apply ReLU as activation function.
    if relu:
      output = tf.nn.relu(output)

  return output

def _pool(x,
          name,
          kernel_size,
          strides,
          padding,
          pool_fn):
  """Helper function for spatial pooling layer.

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels].
    name: The prefix of tensorflow variables defined in this layer.
    kernel_size: A number indicating the size of pooling kernels.
    strides: A number indicating the stride of the sliding window for
      height and width.
    padding: 'VALID' or 'SAME'.
    pool_fn: A tensorflow operation for pooling, such as tf.nn.max_pool.

  Returns:
    A tensor of size [batch_size, height_out, width_out, channels].
  """
  k = kernel_size
  s = strides
  if s > 1 and padding != 'SAME':
    pad = k - 1
    pad_beg = pad // 2
    pad_end = pad - pad_beg
    pad_h = [pad_beg, pad_end]
    pad_w = [pad_beg, pad_end]
    x = tf.pad(x, [[0,0], pad_h, pad_w, [0,0]])


  output = pool_fn(x,
                   ksize=[1, k, k, 1],
                   strides=[1, s, s, 1],
                   padding=padding,
                   name=name)

  return output

def max_pool(x,
             name,
             kernel_size,
             strides,
             padding):
  """Max pooling layer.

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels].
    name: The prefix of tensorflow variables defined in this layer.
    kernel_size: A number indicating the size of pooling kernels.
    strides: A number indicating the stride of the sliding window for
      height and width.
    padding: 'VALID' or 'SAME'.

  Returns:
    A tensor of size [batch_size, height_out, width_out, channels].
  """
  return _pool(x, name, kernel_size, strides, padding, tf.nn.max_pool)

def avg_pool(x, name, kernel_size, strides, padding):
  """Average pooling layer.

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels].
    name: The prefix of tensorflow variables defined in this layer.
    kernel_size: A number indicating the size of pooling kernels.
    strides: A number indicating the stride of the sliding window for
      height and width.
    padding: 'VALID' or 'SAME'.

  Returns:
    A tensor of size [batch_size, height_out, width_out, channels].
  """
  return _pool(x, name, kernel_size, strides, padding, tf.nn.avg_pool)
