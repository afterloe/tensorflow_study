#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras import optimizers

from header import DatasetLoader
from image_to_array_preprocessor import ImageToArrayPreprocessor
from simple_preprocessor import SimplePreprocessor
from simple_dataset_loader import SimpleDatasetLoader
from shallow_network import ShallowNet, show_in_plt


def main():
    from imutils import paths
#    import os
#    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("[info] loading images ... ...")
    imagePaths = list(paths.list_images(args["dataset"]))
    dl: DatasetLoader = SimpleDatasetLoader(preprocessors=[SimplePreprocessor(32, 32), ImageToArrayPreprocessor()])
    data, labels = dl.load(imagePaths, verbose=500)
    data = data.astype("float") / 255.0
    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)
    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)
    print("[info] compiling model ... ...")
    opt = optimizers.gradient_descent_v2.SGD(learning_rate=0.005)
    model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    print("[info] training network ... ...")
    H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)
    print("[info] evaluating network ... ...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat", "dog", "panda"]))
    show_in_plt(H)


if "__main__" == __name__:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset.")
    args = vars(ap.parse_args())
    main()
