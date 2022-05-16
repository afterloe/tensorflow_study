#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from json import loads
from keras.backend import set_value, get_value
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import gradient_descent_v2
from keras.models import load_model
from sys import setrecursionlimit

from callbacks import EpochCheckpoint, TrainingMonitor
from config import tiny_imagenet_config as cfg
from hdf5io import HDF5DatasetGenerator
from preprocessors import SimplePreprocessor, MeanPreprocessor, ImageToArrayPreprocessor
from resnet import ResNet

import matplotlib


matplotlib.use("agg")
setrecursionlimit(5000)

if "__main__" == __name__:
    from argparse import ArgumentParser
    r"""
        python train.py -c D:\Datasets\output\resnet\tinyimagenet 
    """
    ap = ArgumentParser()
    ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
    ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")
    ap.add_argument("-s", "--start_epoch", type=int, default=0, help="epoch to restart training at")
    args = vars(ap.parse_args())


aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
means = loads(open(cfg.DATASET_MEAN).read())

sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()
batchSize: int = 64

trainGenerator = HDF5DatasetGenerator(cfg.TRAIN_HDF5, batchSize, aug=aug, preprocessors=[sp, mp, iap], classes=cfg.NUM_CLASSES)
valGenerator = HDF5DatasetGenerator(cfg.VAL_HDF5, batchSize, preprocessors=[sp, mp, iap], classes=cfg.NUM_CLASSES)

if args["model"] is None:
    print("[info] compiling model")
    model = ResNet.build(64, 64, 3, cfg.NUM_CLASSES, (3, 4, 6), (64, 128, 256, 512), reg=0.0005, dataset="tiny_imagenet")
    opt = gradient_descent_v2.SGD(learning_rate=1e-1, momentum=0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
else:
    print("[info] loading model from %s" % args["model"])
    model = load_model(args["model"])
    print("[info] old learning rate: {}".format(get_value(model.optimizer.lr)))
    set_value(model.optimizer.lr, 1e-5)
    print("[info] new learning rate: {}".format(get_value(model.optimizer.lr)))

callbacks = [
    EpochCheckpoint(args["checkpoints"], every=5, startAt=args["start_epoch"]),
    TrainingMonitor(cfg.FIG_PATH, jsonPath=cfg.JSON_PATH, startAt=args["start_epoch"])
]

model.fit(
    trainGenerator.generator(),
    steps_per_epoch=trainGenerator.numImages // batchSize,
    validation_data=valGenerator.generator(),
    validation_steps=valGenerator.numImages // batchSize,
    epochs=50,
    max_queue_size=batchSize * 2,
    callbacks=callbacks,
    verbose=1
)

trainGenerator.close()
valGenerator.close()
