import os

from PIL import Image
import numpy as np
import scipy.io
import tensorflow as tf

LABEL_COLORS = scipy.io.loadmat('misc/colormapvoc.mat')['colormapvoc']
LABEL_COLORS *= 255
LABEL_COLORS = LABEL_COLORS.astype(np.uint8)
                

def decode_labels(labels, num_classes=21):
  """Encodes label indices to color maps.
    
  Args:
    labels: A tensor of size [batch_size, height_in, width_in, 1]
    num_classes: A number indicating number of valid classes.
    
  Returns:
    A tensor of size [batch_size, height_in, width_in, 3]
  """
  n, h, w, c = labels.shape
  outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
  for i in range(n):
    outputs[i] = LABEL_COLORS[labels[i,:,:,0]]

  return outputs


def inv_preprocess(imgs, img_mean):
  """Inverses image preprocessing of the input images. 
  
  This function adds back the mean vector and convert BGR to RGB.
       
  Args:
    imgs: A tensor of size [batch_size, height_in, width_in, 3]
    img_mean: A 1-D tensor indicating the vector of mean colour values.
  
  Returns:
    A tensor of size [batch_size, height_in, width_in, 3]
  """
  n, h, w, c = imgs.shape
  outputs = np.zeros((n, h, w, c), dtype=np.uint8)
  for i in range(n):
    outputs[i] = (imgs[i] + img_mean).astype(np.uint8)

  return outputs


def snapshot_arg(args):
  """Print and snapshots Command-Line arguments to a text file.
  """
  snap_dir = args.snapshot_dir
  dictargs = vars(args)
  if not os.path.isdir(snap_dir):
    os.makedirs(snap_dir)
  print('-----------------------------------------------')
  print('-----------------------------------------------')
  with open(os.path.join(snap_dir, 'config'), 'w') as argsfile:
    for key, val in dictargs.items():
      line = '| {0} = {1}'.format(key, val)
      print(line)
      argsfile.write(line+'\n')
  print('-----------------------------------------------')
  print('-----------------------------------------------')
