#!/usr/bin/env python3
# -*- coding=utf-8 -*-

# 数据集位置
IMAGES_PATH: str = r"D:\Datasets\dogs-vs-cats\train"
ROOT: str = r"D:\Datasets\output\kaggle_dogs_vs_cats"

# 分类数量
NUM_CLASSES: int = 2
NUM_VAL_IMAGES: int = 1250 * NUM_CLASSES
NUM_TEST_IMAGES: int = 1250 * NUM_CLASSES

# 导出位置
TRAIN_HDF5: str = r"%s\hdf5\train.hdf5" % ROOT
VAL_HDF5: str = r"%s\hdf5\val.hdf5" % ROOT
TEST_HDF5: str = r"%s\hdf5\test.hdf5" % ROOT

# 模型位置
MODEL_PATH: str = r"%s\alexnet_dogs_vs_cats.model" % ROOT
DATASET_MEAN: str = r"%s\dogs_vs_cats_mean.json" % ROOT
OUTPUT_PATH: str = r"%s" % ROOT
