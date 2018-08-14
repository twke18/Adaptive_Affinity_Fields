import os
import argparse

import numpy as np
import PIL.Image as Image


parser = argparse.ArgumentParser(
  description='Transfer TrainId To LabelId For Cityscapes')
parser.add_argument('--pred-dir', type=str, default='',
                    help='/path/to/segment/predictions.')
parser.add_argument('--save-dir', type=str, default='',
                    help='/path/to/saved/results.')
args = parser.parse_args()

TRANSFORM_TB = [
  7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
  23, 24, 25, 26, 27, 28, 31, 32, 33
]

if not os.path.isdir(args.save_dir):
  os.makedirs(args.save_dir)
for dirpath, dirnames, filenames in os.walk(args.pred_dir):
  for filename in filenames:
    predname = os.path.join(dirpath, filename)
    pred = np.asarray(
        Image.open(predname).convert(mode='L'),
        dtype=np.uint8)
    
    new_predname = predname.replace(args.pred_dir,
                                    args.save_dir)
    if not os.path.isdir(os.path.dirname(new_predname)):
      os.makedirs(os.path.dirname(new_predname))

    new_pred = np.zeros_like(pred)
    for train_id, label_id in enumerate(TRANSFORM_TB):
      new_pred[pred == train_id] = label_id

    Image.fromarray(new_pred, mode='L').save(new_predname)
