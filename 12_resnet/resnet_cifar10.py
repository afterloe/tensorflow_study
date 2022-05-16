#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from idna import valid_contextj
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.models import load_model
from keras.backend import get_value, set_value
from keras.optimizers import gradient_descent_v2
from numpy import mean
from os import sep
from sklearn.preprocessing import LabelBinarizer
from sys import setrecursionlimit

from callbacks import EpochCheckpoint, TrainingMonitor
from resnet import ResNet

import matplotlib

matplotlib.use("agg")
setrecursionlimit(5000)

if "__main__" == __name__:
    from argparse import ArgumentParser
    r"""
        python resnet_cifar10.py -o D:\Datasets\output\resnet -c D:\Datasets\output\resnet\checkpoints
        python resnet_cifar10.py -o D:\Datasets\output\resnet -c D:\Datasets\output\resnet\checkpoints -m D:\Datasets\output\resnet\checkpoints\epoch_6.hdf5 -s 6
    """
    ap = ArgumentParser()
    ap.add_argument("-o", "--output", required=True,
                    help="path to output .etc file")
    ap.add_argument("-c", "--checkpoints", required=True,
                    help="path to output checkpoint directroy")
    ap.add_argument("-m", "--model", type=str,
                    help="path to *specifi* model checkpoint to load")
    ap.add_argument("-s", "--start_epoch", type=int, default=0,
                    help="epoch to restart training at")
    args = vars(ap.parse_args())


print("info: loading cifar-10 data")
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

if args["model"] is None:
    print("info: compiling model")
    opt = gradient_descent_v2.SGD(learning_rate=1e-1)
    model = ResNet.build(32, 32, 3, 10, (9, 9, 9),
                         (64, 64, 128, 128), reg=0.0005)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt, metrics=["accuracy"])
else:
    print("info: loading model from %s" % args["model"])
    model = load_model(args["model"])
    print("info: old learning rate: {}".format(get_value(model.optimizer.lr)))
    set_value(model.optimizer.lr, 1e-5)
    print("info: new learning rate: {}".format(get_value(model.optimizer.lr)))

callbacks = [EpochCheckpoint(args["checkpoints"], every=6, startAt=args["start_epoch"]),
             TrainingMonitor(sep.join([args["output"], "resnet56_cifar10.png"]), sep.join([args["output"], "resnet56_cifar10.json"]), startAt=args["start_epoch"])]

print("info: training network")
batchSize: int = 128
model.fit(aug.flow(trainX, trainY, batch_size=batchSize), validation_data=(
    testX, testY), steps_per_epoch=len(trainX) // batchSize, epochs=10, callbacks=callbacks, verbose=1)
print("info: success train")
