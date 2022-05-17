#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from cv2 import imread
from imutils.paths import list_images 
from json import dumps
from numpy import mean
from os import sep
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from progressbar import Percentage, Bar, ETA, ProgressBar

from config import tiny_imagenet_config as cfg
from hdf5io import HDF5DatasetWriter

trainPaths = list(list_images(cfg.TRAIN_IMAGES))
r"""
    tiny-imagenet-200/train/{wordnet_id}/{unique_filename}.JPG
"""
trainLabels = [p.split(sep)[-3] for p in trainPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

dataset = train_test_split(
    trainPaths, trainLabels, test_size=cfg.NUM_TEST_IMAGES, stratify=trainLabels, random_state=42)
trainPaths, testPaths, trainLabels, testLabels = dataset

M = open(cfg.VAL_MAPPINGS).read().strip().split("\n")
M = [r.split("\t")[:2] for r in M]

valPaths = [sep.join([cfg.VAL_IMAGES, m[0]]) for m in M]
valLabels = le.transform([m[1] for m in M])

datasets = [
    ("train", trainPaths, trainLabels, cfg.TRAIN_HDF5),
    ("val", valPaths, valLabels, cfg.VAL_HDF5),
    ("test", testPaths, testLabels, cfg.TEST_HDF5)
]

R, G, B = [], [], []
print("[info] build dataset")
for (dType, paths, labels, outputPath) in datasets:
    print("[info] building %s" % dType)
    writer = HDF5DatasetWriter((len(paths), 64, 64, 3), outputPath)
    widgets = ["Building Dataset: ", Percentage(), " ", Bar(), " ", ETA()]
    pbar = ProgressBar(maxval=len(paths), widgets=widgets)
    pbar.start()
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        image = imread(path)
        if "train" == dType:
            b, g, r = image.shape[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        writer.add([image], [label])
        pbar.update(i)
    pbar.finish()
    writer.close()

print("[info] serializing means")
D = {"R": mean(R), "G": mean(G), "B": mean(B)}
f = open(cfg.DATASET_MEAN)
f.write(dumps(D))
f.close()

print("build hdf5 dataset success")
