#!/usr/bin/env python3
# -*- coding=utf-8 -*-

ROOT: str = r"D:\Datasets\tiny-imagenet-200"
DATASET_ROOT: str = r"D:\Datasets\output\tiny_imagenet_200"

TRAIN_IMAGES: str = r"%s\train" % ROOT
VAL_IMAGES: str = r"%s\val\images" % ROOT

VAL_MAPPINGS: str = r"%s\val\val_annotations.txt" % ROOT
WORDNET_IDS: str = r"%s\wnids.txt" % ROOT
WORD_LABELS: str = r"%s\words.txt" % ROOT

NUM_CLASSES: int = 200
NUM_TEST_IMAGES: int = 50 * NUM_CLASSES

TRAIN_HDF5: str = r"%s\hdf5\train.hdf5" % DATASET_ROOT
VAL_HDF5: str = r"%s\hdf5\val.hdf5" % DATASET_ROOT
TEST_HDF5: str = r"%s\hdf5\test.hdf5" % DATASET_ROOT

DATASET_MEAN = r"%s\tiny-image-net-200-mean.json" % ROOT

OUTPUT_PATH: str = r"D:\Datasets\output\resnet\tinyimagenet"
MODEL_PATH: str = r"%s\model.hdf5" % OUTPUT_PATH
FIG_PATH: str = r"%s\fig.png" % OUTPUT_PATH
JSON_PATH: str = r"%s\log.json" % OUTPUT_PATH