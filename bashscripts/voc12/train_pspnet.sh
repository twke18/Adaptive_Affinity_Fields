#!/bin/bash
# Training Parameters
BATCH_SIZE=8
TRAIN_INPUT_SIZE=336,336
WEIGHT_DECAY=5e-4
ITER_SIZE=1
NUM_STEPS=30000
NUM_CLASSES=21
# Testing Parameters
TEST_INPUT_SIZE=480,480
TEST_STRIDES=320,320
TEST_SPLIT=val
# saved model path
SNAPSHOT_DIR=snapshots/voc12/pspnet/p336_bs8_lr1e-3_it30k
# Procedure pipeline
IS_TRAIN_1=1
IS_TEST_1=1
IS_BENCHMARK_1=1
IS_TRAIN_2=1
IS_TEST_2=1
IS_BENCHMARK_2=1

export PYTHONPATH=`pwd`:$PYTHONPATH
DATAROOT=/path/to/data


# Stage1 Training
if [ ${IS_TRAIN_1} -eq 1 ]; then
  python3 pyscripts/train/train.py\
    --snapshot-dir ${SNAPSHOT_DIR}/stage1\
    --restore-from snapshots/imagenet/trained/resnet_v1_101.ckpt\
    --data-list dataset/voc12/train+.txt\
    --data-dir ${DATAROOT}/VOCdevkit/\
    --batch-size ${BATCH_SIZE}\
    --save-pred-every 10000\
    --update-tb-every 50\
    --input-size ${TRAIN_INPUT_SIZE}\
    --learning-rate 1e-3\
    --weight-decay ${WEIGHT_DECAY}\
    --iter-size ${ITER_SIZE}\
    --num-classes ${NUM_CLASSES}\
    --num-steps $(($NUM_STEPS+1))\
    --random-mirror\
    --random-scale\
    --random-crop\
    --not-restore-classifier\
    --is-training
fi

# Stage1 Testing
if [ ${IS_TEST_1} -eq 1 ]; then
  python3 pyscripts/test/evaluate.py\
    --data-dir ${DATAROOT}/VOCdevkit/\
    --data-list dataset/voc12/${TEST_SPLIT}.txt\
    --input-size ${TEST_INPUT_SIZE}\
    --strides ${TEST_STRIDES}\
    --restore-from ${SNAPSHOT_DIR}/stage1/model.ckpt-${NUM_STEPS}\
    --colormap misc/colormapvoc.mat\
    --num-classes ${NUM_CLASSES}\
    --ignore-label 255\
    --save-dir ${SNAPSHOT_DIR}/stage1/results/${TEST_SPLIT}
fi

if [ ${IS_BENCHMARK_1} -eq 1 ]; then
  python3 pyscripts/utils/benchmark_by_mIoU.py\
    --pred-dir ${SNAPSHOT_DIR}/stage1/results/${TEST_SPLIT}/gray/\
    --gt-dir ${DATAROOT}/VOCdevkit/VOC2012/segcls/\
    --num-classes ${NUM_CLASSES}
fi

# Stage2 Training
if [ ${IS_TRAIN_2} -eq 1 ]; then
  python3 pyscripts/train/train.py\
    --snapshot-dir ${SNAPSHOT_DIR}/stage2\
    --restore-from ${SNAPSHOT_DIR}/stage1/model.ckpt-30000\
    --data-list dataset/voc12/train.txt\
    --data-dir ${DATAROOT}/VOCdevkit/\
    --batch-size ${BATCH_SIZE}\
    --save-pred-every 10000\
    --update-tb-every 50\
    --input-size ${TRAIN_INPUT_SIZE}\
    --learning-rate 1e-4\
    --weight-decay ${WEIGHT_DECAY}\
    --iter-size ${ITER_SIZE}\
    --num-classes ${NUM_CLASSES}\
    --num-steps $(($NUM_STEPS+1))\
    --random-mirror\
    --random-scale\
    --random-crop\
    --is-training
fi

# Stage2 Testing
if [ ${IS_TEST_2} -eq 1 ]; then
  python3 pyscripts/test/evaluate_msc.py\
    --data-dir ${DATAROOT}/VOCdevkit/\
    --data-list dataset/voc12/${TEST_SPLIT}.txt\
    --input-size ${TEST_INPUT_SIZE}\
    --strides ${TEST_STRIDES}\
    --restore-from ${SNAPSHOT_DIR}/stage2/model.ckpt-${NUM_STEPS}\
    --colormap misc/colormapvoc.mat\
    --num-classes ${NUM_CLASSES}\
    --ignore-label 255\
    --flip-aug\
    --scale-aug\
    --save-dir ${SNAPSHOT_DIR}/stage2/results/${TEST_SPLIT}
fi

if [ ${IS_BENCHMARK_2} -eq 1 ]; then
  python3 pyscripts/utils/benchmark_by_mIoU.py\
    --pred-dir ${SNAPSHOT_DIR}/stage2/results/${TEST_SPLIT}/gray/\
    --gt-dir ${DATAROOT}/VOCdevkit/VOC2012/segcls/\
    --num-classes ${NUM_CLASSES}
fi
