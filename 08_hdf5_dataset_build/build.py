#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import json

from imutils.paths import list_images
from os import sep
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import progressbar
from cv2 import imread, mean
from numpy import mean as m

from aspect_aware_preprocessor import AspectAwarePreprocessor
from hdf5_dataset_writer import HDF5DatasetWriter


def main():
    trainPaths = list(list_images(cfg.IMAGES_PATH))
    trainLabels = [p.split(sep)[-1].split(".")[0] for p in trainPaths]
    r"""
    eg:
        D:\Datasets\dogs-vs-cats\train\dog.11046.jpg
        dog
    """
    le = LabelEncoder()
    trainLabels = le.fit_transform(trainLabels)
    data = train_test_split(trainPaths, trainLabels, test_size=cfg.NUM_TEST_IMAGES, stratify=trainLabels,
                            random_state=42)
    trainPaths, testPaths, trainLabels, testLabels = data
    data = train_test_split(trainPaths, trainLabels, test_size=cfg.NUM_VAL_IMAGES, stratify=trainLabels,
                            random_state=42)
    trainPaths, valPaths, trainLabels, valLabels = data
    datasets = [
        ("train", trainPaths, trainLabels, cfg.TRAIN_HDF5),
        ("val", valPaths, valLabels, cfg.VAL_HDF5),
        ("test", testPaths, testLabels, cfg.TEST_HDF5)
    ]
    aap = AspectAwarePreprocessor(256, 256)
    R, G, B = [], [], []
    for (dType, paths, labels, outputPath) in datasets:
        print("[info] building %s dataset" % dType)
        writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath)
        widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        bar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets)
        bar.start()
        for (i, (path, label)) in enumerate(zip(paths, labels)):
            image = imread(path)
            image = aap.preprocess(image)
            if "train" == dType:
                b, g, r = mean(image)[:3]
                R.append(r)
                G.append(g)
                B.append(b)
            writer.add([image], [label])
            bar.update(i)
        bar.finish()
        writer.close()
    print("[info] serializing means ...")
    D = {"R": m(R), "G": m(G), "B": m(B)}
    f = open(cfg.DATASET_MEAN, "w")
    f.write(json.dumps(D))
    f.close()
    print("[info] build success")


if "__main__" == __name__:
    import config as cfg
    main()
