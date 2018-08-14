import os
import argparse

from PIL import Image
import numpy as np

from utils.metrics import iou_stats


parser = argparse.ArgumentParser(
  description='Benchmark segmentation predictions'
)
parser.add_argument('--pred-dir', type=str, default='',
                    help='/path/to/prediction.')
parser.add_argument('--gt-dir', type=str, default='',
                    help='/path/to/ground-truths')
parser.add_argument('--inst-dir', type=str, default='',
                    help='/path/to/instance-mask')
parser.add_argument('--num-classes', type=int, default=21,
                    help='number of segmentation classes')
parser.add_argument('--string-replace', type=str, default=',',
                    help='replace the first string with the second one')
args = parser.parse_args()


assert(os.path.isdir(args.pred_dir))
assert(os.path.isdir(args.gt_dir))
iou = np.zeros(args.num_classes, dtype=np.float64)
ninst = np.zeros(args.num_classes, dtype=np.float64)
for dirpath, dirnames, filenames in os.walk(args.pred_dir):
  for filename in filenames:
    predname = os.path.join(dirpath, filename)
    gtname = predname.replace(args.pred_dir, args.gt_dir)
    instname = predname.replace(args.pred_dir, args.inst_dir)
    if args.string_replace != '':
      stra, strb = args.string_replace.split(',')
      gtname = gtname.replace(stra, strb)
      instname = instname.replace(stra, strb)

    pred = np.asarray(
        Image.open(predname).convert(mode='L'),
        dtype=np.uint8)
    gt = np.asarray(
        Image.open(gtname).convert(mode='L'),
        dtype=np.uint8)
    inst = np.asarray(
        Image.open(instname).convert(mode='P'),
        dtype=np.uint8)

    # Compute true-positive, false-positive
    # and false-negative
    _tp_fn, _tp_fp, _tp = iou_stats(
        pred,
        gt,
        num_classes=args.num_classes,
        background=0)

    # Compute num. of instances per class
    inst_inds = np.unique(inst)
    ninst_ = np.zeros(args.num_classes, dtype=np.float64)
    for i in range(inst_inds.size):
      if i < 255:
        inst_ind = inst_inds[i]
        inst_mask = inst == inst_ind
        seg_mask = gt[inst_mask]
        npixel, _ = np.histogram(
            seg_mask, bins=args.num_classes,
            range=(0, args.num_classes-1)) # num. pixel per class of this inst.
        cls = np.argmax(npixel)
        ninst_[cls] += 1
                              

    iou_ = _tp/(_tp_fn + _tp_fp - _tp + 1e-12)
    iou += iou_*ninst_
    ninst += ninst_

iou /= ninst+1e-12
iou *= 100
class_names = ['Background', 'Aero', 'Bike', 'Bird', 'Boat',
               'Bottle', 'Bus', 'Car', 'Cat', 'Chair','Cow',
               'Table', 'Dog', 'Horse' ,'MBike', 'Person',
               'Plant', 'Sheep', 'Sofa', 'Train', 'TV']

for i in range(args.num_classes):
  print('class {:10s}: {:02d}, acc: {:4.4f}%'.format(
      class_names[i], i, iou[i]))
mean_iou = iou.sum() / args.num_classes
print('mean IOU: {:4.4f}%'.format(mean_iou))
