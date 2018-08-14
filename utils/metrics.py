import numpy as np


def iou_stats(pred, target, num_classes=21, background=0):
  """Computes statistics of true positive (TP), false negative (FN) and
  false positive (FP).

  Args:
    pred: A numpy array.
    target: A numpy array which should be in the same size as pred.
    num_classes: A number indicating the number of valid classes.
    background: A number indicating the class index of the back ground.

  Returns:
    Three num_classes-D vector indicating the statistics of (TP+FN), (TP+FP)
    and TP across each class.
  """
  # Set redundant classes to background.
  locs = np.logical_and(target > -1, target < num_classes)

  # true positive + false negative
  tp_fn, _ = np.histogram(target[locs],
                          bins=np.arange(num_classes+1))
  # true positive + false positive
  tp_fp, _ = np.histogram(pred[locs],
                          bins=np.arange(num_classes+1))
  # true positive
  tp_locs = np.logical_and(locs, pred == target)
  tp, _ = np.histogram(target[tp_locs],
                       bins=np.arange(num_classes+1))

  return tp_fn, tp_fp, tp


def confusion_matrix(pred, target, num_classes=21):
  """Computes the confusion matrix between prediction and ground-truth.

  Args:
    pred: A numpy array.
    target: A numpy array which should be in the same size as pred.
    num_classes: A number indicating the number of valid classes.

  Returns:
    A (num_classes)x(num_classes) 2-D array, in which each row denotes
    ground-truth class, and each column represents predicted class.
  """
  mat = np.zeros((num_classes, num_classes))
  for c in range(num_classes):
    mask = target == c
    if mask.any():
      vec, _ = np.histogram(pred[mask],
                            bins=np.arange(num_classes+1))
      mat[c, :] += vec

  return mat


def accuracy(pred, target):
  """Computes pixel accuracy.

  acc = true_positive / (true_positive + false_positive)

  Args:
    pred: A numpy array.
    target: A numpy array which should be in the same size as pred.

  Returns:
    A number indicating the average accuracy.
  """
  N = pred.shape[0]
  return (pred == target).sum() * 1.0 / N
