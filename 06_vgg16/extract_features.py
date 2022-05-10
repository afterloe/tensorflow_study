#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from imutils.paths import list_images
from keras.applications import imagenet_utils
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array
from numpy import arange, expand_dims, vstack
from os import sep
from progressbar import Percentage, Bar, ETA, ProgressBar
from random import shuffle
from sklearn.preprocessing import LabelEncoder

from hdf5_dataset_writer import HDF5DatasetWriter


def main():
    print("[info] loading images ... ...")
    imagePaths = list(list_images(args["dataset"]))
    shuffle(imagePaths)
    """
    eg:
        dataset_name/{class_label}/example.jpg
    """
    labels = [p.split(sep)[-2] for p in imagePaths]
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    print("[info] loading network ...")
    model = VGG16(weights="imagenet", include_top=False)
    dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7), args["output"], dataKey="features",
                                buffSize=args["buffer_size"])
    dataset.storeClassLabels(le.classes_)
    widgets = ["Extracting Features: ", Percentage(), " ", Bar(), " ", ETA()]
    pbar = ProgressBar(maxval=len(imagePaths), widgets=widgets)
    pbar.start()

    bs = args["batch_size"]
    for i in arange(0, len(imagePaths), bs):
        batchPaths = imagePaths[i: i + bs]
        batchLabels = labels[i: i + bs]
        batchImages = []
        for (j, imagePath) in enumerate(batchPaths):
            try:
                image = load_img(imagePath, target_size=(224, 224))
                image = img_to_array(image)
                image = expand_dims(image, axis=0)
                image = imagenet_utils.preprocess_input(image)
                batchImages.append(image)
            except OSError as e:
                print("[error] %s " % e)
                print("[error] handler %s " % imagePath)
        batchImages = vstack(batchImages)
        features = model.predict(batchImages, batch_size=bs)
        features = features.reshape((features.shape[0], 512 * 7 * 7))
        dataset.add(features, batchLabels)
        pbar.update(i)

    dataset.close()
    pbar.finish()
    pass


if "__main__" == __name__:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-o", "--output", required=True, help="path to output HDF5 file")
    ap.add_argument("-b", "--batch_size", type=int, default=32,
                    help="batch size of images to be passed through network")
    ap.add_argument("-s", "--buffer_size", type=int, default=1000, help="size of feature extraction buffer")
    args = vars(ap.parse_args())
    main()
