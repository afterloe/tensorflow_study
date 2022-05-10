#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from imutils.paths import list_images
from numpy import unique
from os import sep
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.python.keras.optimizers import gradient_descent_v2

from aspect_aware_preprocessor import AspectAwarePreprocessor
from image_to_array_preprocessor import ImageToArrayPreprocessor
from minivgg_network import MiniVGGNet
from simple_dataset_loader import SimpleDatasetLoader


def show_plt(H, epochs: int = 100):
    import matplotlib.pyplot as plt
    from numpy import arange
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(arange(0, epochs), H.history["accuracy"], label="accuracy")
    plt.plot(arange(0, epochs), H.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


def main():
    print("[info] loading images ...")
    imagePaths = list(list_images(args["dataset"]))
    """
    eg:
        flowers17/{species}/{image}
        flowers17/bluebell/image_0241.jpg
    """
    classNames = [p.split(sep)[-2] for p in imagePaths]
    classNames = [str(x) for x in unique(classNames)]
    aap = AspectAwarePreprocessor(64, 64)
    iap = ImageToArrayPreprocessor()
    sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
    data, labels = sdl.load(imagePaths, verbose=500)
    data = data.astype("float") / 255.0
    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)

    print("[info] compiling model ...")
    # opt = gradient_descent_v2.SGD(learning_rate=0.05, decay=0.01 / 40, momentum=0.9, nesterov=True)
    opt = gradient_descent_v2.SGD(learning_rate=0.05)
    model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(classNames))
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("[info] training network ...")
    H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)

    print("[info] evaluating network ...")
    predictions = model.predict(testX, batch_size=32)

    print("[info] network report")
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))
    show_plt(H, epochs=100)


if "__main__" == __name__:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    args = vars(ap.parse_args())
    main()
