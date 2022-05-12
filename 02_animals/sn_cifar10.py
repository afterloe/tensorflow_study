#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.datasets import cifar10
from keras.optimizers import gradient_descent_v2

from shallow_network import ShallowNet, show_in_plt


def main():
    print("[info] loading CIFAR-10 dataset ... ...")
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainX = trainX.astype("float") / 255.0
    testX = testX.astype("float") / 255.0

    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)

    labelNames = ["airplane", "automobile", "bird", "cat",
                  "deer", "dog", "frog", "horse", "ship", "truck"]
    print("[info] compiling model ... ...")
    opt = gradient_descent_v2.SGD(learning_rate=0.01)
    model = ShallowNet.build(32, 32, 3, classes=10)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt, metrics=["accuracy"])
    print("[info] training network ... ...")
    H = model.fit(trainX, trainY, validation_data=(
        testX, testY), batch_size=32, epochs=40, verbose=1)
    print("[info] evaluating network ... ...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
          predictions.argmax(axis=1), target_names=labelNames))
    show_in_plt(H, value=40)


if "__main__" == __name__:
    main()
