#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from cv2 import imread, putText, FONT_HERSHEY_SIMPLEX, imshow, waitKey, destroyAllWindows
from numpy import array
from imutils.paths import list_images
from keras.models import load_model

from header import DatasetLoader
from simple_dataset_loader import SimpleDatasetLoader
from simple_preprocessor import SimplePreprocessor
from image_to_array_preprocessor import ImageToArrayPreprocessor


def main():
    classLabels = ["cat", "dog", "panda"]
    print("[info] sampling images ...")
    imagePaths = array(list(list_images(args["dataset"])))
    dl: DatasetLoader = SimpleDatasetLoader(preprocessors=[SimplePreprocessor(32, 32), ImageToArrayPreprocessor()])
    data, labels = dl.load(imagePaths)
    data = data.astype("float") / 255.0
    print("[info] loading pre-trained network ...")
    model = load_model(args["model"])
    print("[info] predicting ...")
    predictions = model.predict(data, batch_size=32).argmax(axis=1)
    for (i, imagePath) in enumerate(imagePaths):
        image = imread(imagePath)
        putText(image, "maybe is %s" % classLabels[predictions[i]], (10, 30), FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        imshow("Image", image)
        waitKey(0)
    destroyAllWindows()


if "__main__" == __name__:
    """
    cmd
    conda activate opecv-base
    python sn_load.py -d D:\Datasets\animals\sample -m D:\Datasets\output\shallownet_weights.hdf5
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
    args = vars(ap.parse_args())
    main()
