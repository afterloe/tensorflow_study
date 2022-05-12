#!/usr/bin/env python3
# -*- coding=utf-8 -*-

ROOT: str = r"D:\Datasets\tiny-imagenet-200"
DATASET_ROOT: str = r"D:\Datasets\output\tiny_imagenet_200"

TRAIN_IMAGES: str = r"%s\train" % ROOT
VAL_IMAGES: str = r"%s\val\images" % ROOT

VAL_MAPPINGS: str = r"%s\val\val_annotations.txt" % ROOT
WORDNET_IDS: str = r"%s\wnids.txt" % ROOT
WORD_LABELS: str = r"%S\words.txt" % ROOT

NUM_CLASSES: int = 200
NUM_TEST_IMAGES: int = 50 * NUM_CLASSES

TRAIN_HDF5: str = r"%s\hdf5\train.hdf5" % DATASET_ROOT
VAL_HDF5: str = r"%S\hdf5\val.hdf5" % DATASET_ROOT
TEST_HDF5: str = r"%S\hdf5\test.hdf5" % DATASET_ROOT

DATASET_MEAN = r"%s\tiny-image-net-200-mean.json" % ROOT

MODEL_PATH = r"%s\epoch_70.hdf5" % ROOT
FIG_PATH = r"%S\deepergooglenet_tinyimagenet.png" % ROOT
JSON_PATH = r"%s\deepergooglenet_tinyimagenet.json" % ROOT
