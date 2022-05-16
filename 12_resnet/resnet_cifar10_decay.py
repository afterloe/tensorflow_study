#!/usr/bin/env python3
# -*- coding=utf-8 -*-


from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import gradient_descent_v2
from keras.datasets import cifar10
from numpy import mean
from os import sep, getpid
from sys import setrecursionlimit
from sklearn.preprocessing import LabelBinarizer

from callbacks import TrainingMonitor
from resnet import ResNet

import matplotlib
matplotlib.use("agg")
setrecursionlimit(5000)

if "__main__" == __name__:
    from argparse import ArgumentParser
    r"""
        python resnet_cifar10_decay.py -m D:\Datasets\output\resnet\resnet_cifar10.hdf5 -o D:\Datasets\output\resnet
    """
    ap = ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to output model")
    ap.add_argument("-o", "--output", required=True,
                    help="path to output directory (logs, plots, etc.)")
    args = vars(ap.parse_args())


NUM_EPOCHS: int = 100
INIT_LR: float = 1e-1


def poly_decay(epoch: int):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    return alpha


print("[info] loading CIFAR-10 data")
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

means = mean(trainX, axis=0)
trainX -= means
testX -= means

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                         horizontal_flip=True, fill_mode="nearest")
figPath = sep.join([args["output"], "%d.png" % getpid()])
jsonPath = sep.join([args["output"], "%d.json" % getpid()])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath),
             LearningRateScheduler(poly_decay)]

print("[info] compiling model")
opt = gradient_descent_v2.SGD(learning_rate=INIT_LR, momentum=0.9)
model = ResNet.build(32, 32, 3, 10, (9, 9, 9), (64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

print("[info] training network")
batchSize: int = 128
model.fit(aug.flow(trainX, trainY, batch_size=batchSize), validation_data=(
    testX, testY), steps_per_epoch=len(trainX)//128, epochs=10, callbacks=callbacks, verbose=1)

print("[info] serializing newtwok")
model.save(args["model"])
