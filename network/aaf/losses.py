import tensorflow as tf

import network.aaf.layers as nnx


def affinity_loss(labels,
                  probs,
                  num_classes,
                  kld_margin):
  """Affinity Field (AFF) loss.

  This function computes AFF loss. There are several components in the
  function:
  1) extracts edges from the ground-truth labels.
  2) extracts ignored pixels and their paired pixels (the neighboring
     pixels on the eight corners).
  3) extracts neighboring pixels on the eight corners from a 3x3 patch.
  4) computes KL-Divergence between center pixels and their neighboring
     pixels from the eight corners.

  Args:
    labels: A tensor of size [batch_size, height_in, width_in], indicating 
      semantic segmentation ground-truth labels.
    probs: A tensor of size [batch_size, height_in, width_in, num_classes],
      indicating segmentation predictions.
    num_classes: A number indicating the total number of valid classes.
    kld_margin: A number indicating the margin for KL-Divergence at edge.

  Returns:
    Two 1-D tensors value indicating the loss at edge and non-edge.
  """
  # Compute ignore map (e.g, label of 255 and their paired pixels).
  labels = tf.squeeze(labels, axis=-1) # NxHxW
  ignore = nnx.ignores_from_label(labels, num_classes, 1) # NxHxWx8
  not_ignore = tf.logical_not(ignore)
  not_ignore = tf.expand_dims(not_ignore, axis=3) # NxHxWx1x8

  # Compute edge map.
  one_hot_lab = tf.one_hot(labels, depth=num_classes)
  edge = nnx.edges_from_label(one_hot_lab, 1, 255) # NxHxWxCx8

  # Remove ignored pixels from the edge/non-edge.
  edge = tf.logical_and(edge, not_ignore)
  not_edge = tf.logical_and(tf.logical_not(edge), not_ignore)

  edge_indices = tf.where(tf.reshape(edge, [-1]))
  not_edge_indices = tf.where(tf.reshape(not_edge, [-1]))

  # Extract eight corner from the center in a patch as paired pixels.
  probs_paired = nnx.eightcorner_activation(probs, 1)  # NxHxWxCx8
  probs = tf.expand_dims(probs, axis=-1) # NxHxWxCx1
  bot_epsilon = tf.constant(1e-4, name='bot_epsilon')
  top_epsilon = tf.constant(1.0, name='top_epsilon')
  neg_probs = tf.clip_by_value(
      1-probs, bot_epsilon, top_epsilon)
  probs = tf.clip_by_value(
      probs, bot_epsilon, top_epsilon)
  neg_probs_paired= tf.clip_by_value(
      1-probs_paired, bot_epsilon, top_epsilon)
  probs_paired = tf.clip_by_value(
    probs_paired, bot_epsilon, top_epsilon)

  # Compute KL-Divergence.
  kldiv = probs_paired*tf.log(probs_paired/probs)
  kldiv += neg_probs_paired*tf.log(neg_probs_paired/neg_probs)
  not_edge_loss = kldiv
  edge_loss = tf.maximum(0.0, kld_margin-kldiv)

  not_edge_loss = tf.reshape(not_edge_loss, [-1])
  not_edge_loss = tf.gather(not_edge_loss, not_edge_indices)
  edge_loss = tf.reshape(edge_loss, [-1])
  edge_loss = tf.gather(edge_loss, edge_indices)

  return edge_loss, not_edge_loss


def adaptive_affinity_loss(labels,
                           one_hot_lab,
                           probs,
                           size,
                           num_classes,
                           kld_margin,
                           w_edge,
                           w_not_edge):
  """Adaptive affinity field (AAF) loss.

  This function computes AAF loss. There are three components in the function:
  1) extracts edges from the ground-truth labels.
  2) extracts ignored pixels and their paired pixels (usually the eight corner
     pixels).
  3) extracts eight corner pixels/predictions from the center in a
     (2*size+1)x(2*size+1) patch
  4) computes KL-Divergence between center pixels and their paired pixels (the 
     eight corner).
  5) imposes adaptive weightings on the loss.

  Args:
    labels: A tensor of size [batch_size, height_in, width_in], indicating 
      semantic segmentation ground-truth labels.
    one_hot_lab: A tensor of size [batch_size, height_in, width_in, num_classes]
      which is the ground-truth labels in the form of one-hot vector.
    probs: A tensor of size [batch_size, height_in, width_in, num_classes],
      indicating segmentation predictions.
    size: A number indicating the half size of a patch.
    num_classes: A number indicating the total number of valid classes. The 
    kld_margin: A number indicating the margin for KL-Divergence at edge.
    w_edge: A number indicating the weighting for KL-Divergence at edge.
    w_not_edge: A number indicating the weighting for KL-Divergence at non-edge.

  Returns:
    Two 1-D tensors value indicating the loss at edge and non-edge.
  """
  # Compute ignore map (e.g, label of 255 and their paired pixels).
  labels = tf.squeeze(labels, axis=-1) # NxHxW
  ignore = nnx.ignores_from_label(labels, num_classes, size) # NxHxWx8
  not_ignore = tf.logical_not(ignore)
  not_ignore = tf.expand_dims(not_ignore, axis=3) # NxHxWx1x8

  # Compute edge map.
  edge = nnx.edges_from_label(one_hot_lab, size, 255) # NxHxWxCx8

  # Remove ignored pixels from the edge/non-edge.
  edge = tf.logical_and(edge, not_ignore)
  not_edge = tf.logical_and(tf.logical_not(edge), not_ignore)

  edge_indices = tf.where(tf.reshape(edge, [-1]))
  not_edge_indices = tf.where(tf.reshape(not_edge, [-1]))

  # Extract eight corner from the center in a patch as paired pixels.
  probs_paired = nnx.eightcorner_activation(probs, size)  # NxHxWxCx8
  probs = tf.expand_dims(probs, axis=-1) # NxHxWxCx1
  bot_epsilon = tf.constant(1e-4, name='bot_epsilon')
  top_epsilon = tf.constant(1.0, name='top_epsilon')

  neg_probs = tf.clip_by_value(
      1-probs, bot_epsilon, top_epsilon)
  neg_probs_paired = tf.clip_by_value(
      1-probs_paired, bot_epsilon, top_epsilon)
  probs = tf.clip_by_value(
      probs, bot_epsilon, top_epsilon)
  probs_paired = tf.clip_by_value(
    probs_paired, bot_epsilon, top_epsilon)

  # Compute KL-Divergence.
  kldiv = probs_paired*tf.log(probs_paired/probs)
  kldiv += neg_probs_paired*tf.log(neg_probs_paired/neg_probs)
  edge_loss = tf.maximum(0.0, kld_margin-kldiv)
  not_edge_loss = kldiv

  # Impose weights on edge/non-edge losses.
  one_hot_lab = tf.expand_dims(one_hot_lab, axis=-1)
  w_edge = tf.reduce_sum(w_edge*one_hot_lab, axis=3, keep_dims=True) # NxHxWx1x1
  w_not_edge = tf.reduce_sum(w_not_edge*one_hot_lab, axis=3, keep_dims=True) # NxHxWx1x1

  edge_loss *= w_edge
  not_edge_loss *= w_not_edge

  not_edge_loss = tf.reshape(not_edge_loss, [-1])
  not_edge_loss = tf.gather(not_edge_loss, not_edge_indices)
  edge_loss = tf.reshape(edge_loss, [-1])
  edge_loss = tf.gather(edge_loss, edge_indices)

  return edge_loss, not_edge_loss
