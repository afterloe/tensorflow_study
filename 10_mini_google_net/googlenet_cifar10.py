#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import matplotlib

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import gradient_descent_v2
from sklearn.preprocessing import LabelBinarizer
from numpy import mean
from os import sep, getpid

from training_monitor import TrainingMonitor
from minig_google_network import MiniGoogleNet


def poly_decay(epoch: int):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    return alpha


def main():
    matplotlib.use("agg")
    print("[info] loading CIFAR-10 data")
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainX = trainX.astype("float")
    testX = testX.astype("float")

    m = mean(trainX, axis=0)
    trainX -= m
    testX -= m

    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)

    aug = ImageDataGenerator(
        width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")
    figPath = sep.join([args["output"], "googlenet_cifar10_%d.png" % getpid()])
    jsonPath = sep.join(
        [args["output"], "googlenet_cifar10_%d.json" % getpid()])
    callbacks = [TrainingMonitor(
        figPath, jsonPath=jsonPath), LearningRateScheduler(poly_decay)]
    print("[info] compiling model")
    opt = gradient_descent_v2.SGD(learning_rate=INIT_LR, momentum=0.9)
    model = MiniGoogleNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt, metrics=["accuracy"])

    print("[info] training network")
    batchSize: int = 64
    model.fit(aug.flow(trainX, trainY, batch_size=batchSize), validation_data=(testX, testY), steps_per_epoch=len(
        trainX) // batchSize, epochs=NUM_EPOCHS, callbacks=callbacks, verbose=1)

    print("[info] serializing network")
    model.save(args["model"])


if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to output model")
    ap.add_argument("-o", "--output", required=True,
                    help="path to output directory (logs, plots, etc.)")
    args = vars(ap.parse_args())
    NUM_EPOCHS: int = 70
    INIT_LR: float = 5e-3
    main()
