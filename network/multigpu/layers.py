import math

import tensorflow as tf
import numpy as np

import network.common.layers as nn
from network.multigpu.utils import on_each_gpu

DEFAULT_DEVICE = '/cpu:0'


@on_each_gpu
def ReLU(x):
  """Performs ReLU on each GPU device.
  """
  return tf.nn.relu(x)


@on_each_gpu
def dropout(x, keep_prob=1.0):
  """Performs dropout on each GPU device.
  """
  return tf.nn.dropout(x, keep_prob)


def moments(xs, name, axes):
  """Synchronized computation of batch mean and batch variance across
  multiple GPUs.

  Args:
    xs: A list of tensors that each tensor is on different GPU devices.
    name: A prefix of the variable names defined in this layer.
    axes: A list of numbers indicating the dimensions where mean and variance
      are computed.

  Returns:
    A tensor indicating batch mean and variance.
  """
  n_x = 0.0
  for x in xs:
    shape_x = x.get_shape().as_list()
    n = 1
    for i in axes:
      n *= shape_x[i]
    n_x += n

  # Synchronize the mean.
  means = []
  for x in xs:
    with tf.device(x.device):
      m = tf.reduce_sum(x, axis=axes)
      m /= n_x
      means.append(m)

  batch_mean = tf.add_n(means)

  # Synchronize the variance.
  variances = []
  for x in xs:
    with tf.device(x.device):
      m = tf.reshape(batch_mean, (1,1,1,-1))
      m = tf.stop_gradient(m)
      var = tf.reduce_sum(tf.squared_difference(x, m), axis=axes)
      var /= n_x
      variances.append(var)
  batch_var = tf.add_n(variances)

  return batch_mean, batch_var


def batch_norm(xs,
               name,
               activation_fn=None,
               decay=0.99,
               epsilon=0.001,
               is_training=True):
  """Synchronized multi-gpu batch normalization.

  This function perform batch normalization. If it is set for training,
  it will update moving mean and moving variance to keep track of global
  statistics by exponential decay. The batch mean and batch variance are
  synchronized across all the GPU devices.

  output =  [(x - mean) / sqrt(var)] * gamma + beta.

  Args:
    xs: A list of tensor, where each tensor if  of size
      [batch_size, height_in, width_in, channels].
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
    A list of tensor, where each tensor is  of size
    [batch_size, height_in, width_in, channels]
  """
  shape_x = xs[0].get_shape().as_list()
  dim_x = len(shape_x)
  if dim_x < 2:
    raise ValueError('batch_norm only supports for tensors'
                     +' with dimension larger than 2')
  c_x = shape_x[-1]

  with tf.variable_scope(name) as scope:
    with tf.device(DEFAULT_DEVICE):
      beta = tf.get_variable(
          'beta',
          c_x,
          initializer=tf.constant_initializer(0.0),
          trainable=is_training)
      gamma = tf.get_variable(
          'gamma',
          c_x,
          initializer=tf.constant_initializer(1.0),
          trainable=is_training)
      moving_mean = tf.get_variable(
          'moving_mean',
          c_x,
          initializer=tf.constant_initializer(0.0),
          trainable=False)
      moving_var = tf.get_variable(
          'moving_variance',
          c_x,
          initializer=tf.constant_initializer(1.0),
          trainable=False)

    outputs = []
    if is_training:
      # Update moving mean and variance before applying batch normalization.

      ## step 1, collect mean & var from each mini-batch
      with tf.device(DEFAULT_DEVICE):
        axes = np.arange(dim_x-1)
        batch_mean, batch_var = moments(xs, 'moments', axes)

        # step 2, update moving mean and var operation to update variable by
        # moving average.
        update_moving_mean = tf.assign(
            moving_mean,
            moving_mean*decay + batch_mean*(1-decay))

        update_moving_var = tf.assign(
            moving_var,
            moving_var*decay + batch_var*(1-decay))
        update_ops = [update_moving_mean, update_moving_var]

      # step 3, nomarlize the batch on each GPU device.
      with tf.control_dependencies(update_ops):
        for i,x in enumerate(xs):
          with tf.device(x.device):
            output = tf.nn.batch_normalization(
                x,
                batch_mean,
                batch_var,
                beta,
                gamma,
                epsilon)

            if activation_fn:
              output = activation_fn(output)

            outputs.append(output)

    else:
      for i,x in enumerate(xs):
        with tf.device(x.device):
          output = tf.nn.batch_normalization(
              x,
              moving_mean,
              moving_var,
              beta,
              gamma,
              epsilon)

          if activation_fn:
            output = activation_fn(output)

          outputs.append(output)

  return outputs


def conv(xs,
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
  """Convolutional layers with batch normalization and ReLU on each
  GPU device.

  This function perform convolution, batch_norm (if bn=True),
  and ReLU (if relu=True). The batch normamlization is synchronized
  across all the GPU devices.

  Args:
    xs: A list of tensor, in which each tensor is  of size
      [batch_size, height_in, width_in, channels].
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
    A list of tensor, in which each tensor is of size
    [batch_size, height_out, width_out, channels_out].
  """
  c_i = xs[0].get_shape().as_list()[-1] # input channels
  c_o = filters # output channels

  # Define helper function.
  convolve = lambda i,k: tf.nn.conv2d(
      i,
      k,
      [1, strides, strides, 1],
      padding=padding)

  with tf.variable_scope(name) as scope:
    with tf.device(DEFAULT_DEVICE):
      msra_init = tf.contrib.layers.variance_scaling_initializer(
          2.0, 'FAN_IN', True)
      kernel = tf.get_variable(
          name='weights',
          shape=[kernel_size, kernel_size, c_i, c_o],
          trainable=is_training,
          initializer=msra_init)

      if biased:
        biases = tf.get_variable('biases', [c_o])

    outputs = []
    for x in xs:
      with tf.device(x.device):
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
          output = tf.nn.bias_add(output, biases)

        outputs.append(output)

    # Apply synchronized batch normalization.
    if bn:
      is_bn_training = not use_global_status
      outputs = batch_norm(outputs, 'BatchNorm',
                           is_training=is_bn_training,
                           decay=decay,
                           activation_fn=None)

    # Apply ReLU as activation function.
    if relu:
      outputs = ReLU(outputs)

  return outputs


def atrous_conv(xs,
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
  """Atrous convolutional layers with batch normalization and ReLU on
  each GPU devices.

  This function perform atrous convolution, batch_norm (if bn=True),
  and ReLU (if relu=True). The batch normalization is synchronized
  across all GPU devices.

  Args:
    xs: A list of tensor, in which each tensor is of size
      [batch_size, height_in, width_in, channels].
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
    A list of tensor, in which each tensor is  of size
    [batch_size, height_out, width_out, channels_out].
  """
  c_i = xs[0].get_shape().as_list()[-1] # input channels
  c_o = filters # output channels

  # Define helper function.
  convolve = lambda i,k: tf.nn.atrous_conv2d(
      i,
      k,
      dilation,
      padding)

  with tf.variable_scope(name) as scope:
    with tf.device(DEFAULT_DEVICE):
      kernel = tf.get_variable(
          name='weights',
          shape=[kernel_size, kernel_size, c_i, c_o],
          trainable=is_training,)
      if biased:
        biases = tf.get_variable('biases', [c_o])

    outputs = []
    for x in xs:
      with tf.device(x.device):
        output = convolve(x, kernel)
        # Add the biases.
        if biased:
          output = tf.nn.bias_add(output, biases)

        outputs.append(output)


    # Apply synchronized batch normalization.
    if bn:
      is_bn_training = not use_global_status
      outputs = batch_norm(outputs, 'BatchNorm',
                           is_training=is_bn_training,
                           decay=decay,
                           activation_fn=None)

    # Apply ReLU as activation function.
    if relu:
      outputs = ReLU(outputs)

  return outputs


@on_each_gpu
def max_pool(x, name, kernel_size, strides, padding):
  """Max pooling layer on each GPU device.

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
  return nn.max_pool(x, name, kernel_size, strides, padding)


@on_each_gpu
def avg_pool(xs, name, kernel_size, strides, padding):
  """Average pooling layer on each GPU device.

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
  return nn.avg_pool(x, name, kernel_size, strides, padding)


def split(x, num_gpu):
  """Splits a tensor into sub tensors evenly, and allocate  to each gpu.
  To avoid the case of OOM, the first gpus are allocated with smaller
  mini-batch size.

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels]
    num_gpu: A number indicating the number of GPU devices for parallel
      computation.

  Returns:
    A list of tensor, in which each tensor is allocated on each GPU device.
    The batch size of each tensor is even across all GPU devices, and the 
    tensor is of size [sub_batch_size, height_in, width_in, channels].
  """
  N = x.get_shape().as_list()[0] # real batch_size
  if N < num_gpu:
    raise ValueError('Batch_size is smaller than number of gpu')

  avg_n = int(math.floor(N/num_gpu))
  mod_n = int(N % num_gpu)
  size_splits = [avg_n] * num_gpu
  for i in range(mod_n):
    size_splits[i] += 1 # [avg_n+1, ..., avg_n]
  size_splits = size_splits[::-1] # [avg_n, ..., avg_n+1]
  xs = tf.split(x, size_splits, axis=0)

  # Allocate to each gpu.
  xs_gpu = []
  for i in range(num_gpu):
    gpu_id = '/gpu:{:d}'.format(i)
    with tf.device(gpu_id):
      xs_gpu.append(tf.identity(xs[i]))

  return xs_gpu


def concat(xss, name, axis):
  """Performs tf.concat on each GPU device.
  Args:
    xss: list of list of Tensors. The outer list denotes
         different data Tensors; the inner list denotes
         the same data Tensors on each GPU.

  For example, xss = [
                 [x_gpu_0,...,x_gpu_n],
                 [y_gpu_0,...,y_gpu_n],
                 ...,
                 [z_gpu_0,...,z_gpu_n]
               ]
  outputs = [
    tf.concat([x_gpu_0, y_gpu_0, ..., z_gpu_0]),
    ...,
    tf.concat([x_gpu_n, y_gpu_n, ..., z_gpu_n])
  ]
  """
  num_gpu = len(xss[0])
  # Perform sanity check.
  for xs in xss:
    assert(len(xs) == num_gpu)

  outputs = []
  for i in range(num_gpu):
    with tf.device(xss[0][i].device):
      tensors = [xs[i] for xs in xss]
      cats = tf.concat(tensors, name=name, axis=axis)
      outputs.append(cats)

  return outputs
