#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from imutils.paths import list_images
from random import shuffle
from os import sep
from sklearn.preprocessing import LabelEncoder
from keras.applications.resnet import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from progressbar import Percentage, Bar, ETA, ProgressBar
from numpy import arange, expand_dims, vstack

from hdf5io import HDF5DatasetWriter


def main():
    batchSize: int = args["batch_size"]

    print("[info] loading image")
    imagePaths = list(list_images(args["dataset"]))
    shuffle(imagePaths)
    r"""
    eg:
        D:\Datasets\dogs-vs-cats\train\dog.11046.jpg
        dog
    """
    labels = [p.split(sep)[-1].split(".")[0] for p in imagePaths]
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    print("[info] loading network")
    model = ResNet50(weights="imagenet", include_top=False)
     # print(features.shape) (16 7 7 2048)
    featureSize = 2048 * 7 * 7
    dataset = HDF5DatasetWriter((len(imagePaths), featureSize), args["output"], dataKey="features",
                                buffSize=args["buffer_size"])
    dataset.storeClassLabels(le.classes_)
    widgets = ["Extracting Feature: ", Percentage(), " ", Bar(), " ", ETA()]
    pbar = ProgressBar(maxval=len(imagePaths), widgets=widgets)
    pbar.start()

    for i in arange(0, len(imagePaths), batchSize):
        batchPaths = imagePaths[i: i + batchSize]
        batchLabels = labels[i: i + batchSize]
        batchImages = []
        for (j, imagePath) in enumerate(batchPaths):
            image = load_img(imagePath, target_size=(224, 224))
            image = img_to_array(image)
            image = expand_dims(image, axis=0)
            image = preprocess_input(image)
            batchImages.append(image)
        batchImages = vstack(batchImages)
        features = model.predict(batchImages, batch_size=batchSize)
        features = features.reshape(features.shape[0], featureSize)
        dataset.add(features, batchLabels)
        pbar.update(i)

    dataset.close()
    pbar.finish()


if "__main__" == __name__:
    from argparse import ArgumentParser
    r"""
        python extract_feature.py -d D:\Datasets\dogs-vs-cats\train \
    -o D:\Datasets\output\kaggle_dogs_vs_cats\hdf5\features.hdf5
    """
    ap = ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input src images")
    ap.add_argument("-o", "--output", required=True,
                    help="path ot output HDF5 file")
    ap.add_argument("-b", "--batch_size", type=int, default=16, help="batch size of images to be passed "
                                                                     "through network")
    ap.add_argument("-s", "--buffer_size", type=int, default=1000,
                    help="size of feature extraction buffer")
    args = vars(ap.parse_args())
    main()
