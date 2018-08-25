#!/bin/bash
# This script is used for training, inference and benchmarking
# the baseline method with PSPNet on Cityscapes with multi-gpus.
# Users could also modify from this script for their use case.
#
# Usage:
#   # From Adaptive_Affinity_Fields/ directory.
#   bash bashscripts/cityscapes/train_pspnet_mgpu.sh
#
#

# Set up parameters for training.
BATCH_SIZE=8
TRAIN_INPUT_SIZE=720,720
WEIGHT_DECAY=5e-4
ITER_SIZE=1
NUM_STEPS=90000
NUM_CLASSES=19
NUM_GPU=4

# Set up parameters for inference.
INFERENCE_INPUT_SIZE=720,720
INFERENCE_STRIDES=480,480
INFERENCE_SPLIT=val

# Set up path for saving models.
SNAPSHOT_DIR=snapshots/cityscapes/pspnet/p720_bs8_lr1e-3_it90k

# Set up the procedure pipeline.
IS_TRAIN=1
IS_INFERENCE=1
IS_BENCHMARK=1

# Update PYTHONPATH.
export PYTHONPATH=`pwd`:$PYTHONPATH

# Set up the data directory.
DATAROOT=/path/to/data

# Train.
if [ ${IS_TRAIN} -eq 1 ]; then
  python3 pyscripts/train/train_mgpu.py\
    --snapshot-dir ${SNAPSHOT_DIR}\
    --restore-from snapshots/imagenet/trained/resnet_v1_101.ckpt\
    --data-list dataset/cityscapes/train.txt\
    --data-dir ${DATAROOT}/Cityscapes/\
    --batch-size ${BATCH_SIZE}\
    --save-pred-every ${NUM_STEPS}\
    --update-tb-every 500\
    --input-size ${TRAIN_INPUT_SIZE}\
    --learning-rate 1e-3\
    --weight-decay ${WEIGHT_DECAY}\
    --iter-size ${ITER_SIZE}\
    --num-classes ${NUM_CLASSES}\
    --num-steps $(($NUM_STEPS+1))\
    --num-gpu ${NUM_GPU}\
    --random-mirror\
    --random-scale\
    --random-crop\
    --not-restore-classifier\
    --is-training
fi

# Inference.
if [ ${IS_INFERENCE} -eq 1 ]; then
  python3 pyscripts/inference/inference_msc.py\
    --data-dir ${DATAROOT}/Cityscapes/\
    --data-list dataset/cityscapes/${INFERENCE_SPLIT}.txt\
    --input-size ${INFERENCE_INPUT_SIZE}\
    --strides ${INFERENCE_STRIDES}\
    --restore-from ${SNAPSHOT_DIR}/model.ckpt-${NUM_STEPS}\
    --colormap misc/colormapcs.mat\
    --num-classes ${NUM_CLASSES}\
    --ignore-label 255\
    --flip-aug\
    --scale-aug\
    --save-dir ${SNAPSHOT_DIR}/results/${INFERENCE_SPLIT}
fi

# Benchmark.
if [ ${IS_BENCHMARK} -eq 1 ]; then
  python3 pyscripts/benchmark/benchmark_by_mIoU.py\
    --pred-dir ${SNAPSHOT_DIR}/results/${INFERENCE_SPLIT}/gray/\
    --gt-dir ${DATAROOT}/Cityscapes/gtFineId/${TEST_SPLIT}/all/\
    --num-classes ${NUM_CLASSES}\
    --string-replace leftImg8bit,gtFineId_labelIds
fi
