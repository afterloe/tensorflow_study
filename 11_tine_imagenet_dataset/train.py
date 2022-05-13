#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from json import loads
from keras import backend
from keras.models import load_model
from keras.optimizers import adam_v2
from keras.preprocessing.image import ImageDataGenerator
from config import tiny_imagenet_config as cfg
from preprocessors import ImageToArrayPreprocessor, SimplePreprocessor, MeanPreprocessor
from hdf5io import HDF5DatasetGenerator
from nn.conv import DeeperGoogleNet
from callbacks import TrainingMonitor, EpochCheckpoint
import matplotlib
matplotlib.use("agg")


def main():
    aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15, width_shift_range=0.2,
                             height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
    means = loads(open(cfg.DATASET_MEAN).read())
    sp = SimplePreprocessor(64, 64)
    mp = MeanPreprocessor(means["R"], means["G"], means["B"])
    iap = ImageToArrayPreprocessor()

    batchSize: int = 64

    trainGenerator = HDF5DatasetGenerator(cfg.TRAIN_HDF5, batchSize, aug=aug, preprocessors=[
                                          sp, mp, iap], classes=cfg.NUM_CLASSES)
    valGenerator = HDF5DatasetGenerator(cfg.VAL_HDF5, batchSize, preprocessors=[
                                        sp, mp, iap], classes=cfg.NUM_CLASSES)

    if args["model"] is None:
        print("[info] compiling model")
        model = DeeperGoogleNet.build(
            width=64, height=64, depth=3, classes=cfg.NUM_CLASSES, reg=0.0002)
        opt = adam_v2.Adam(1e-3)
        model.compile(loss="categorical_crossentropy",
                      optimizer=opt, metrics=["accuracy"])
    else:
        print("[info] loading %s" % args["model"])
        model = load_model(args["model"])
        print("[info] old learning rate: {}".format(
            backend.get_value(model.optimizer.lr)))
        backend.set_value(model.optimizer.lr, 1e-5)
        print("[info] new learning rate: {}".format(
            backend.get_value(model.optimizer.lr)))

    callbacks = [
        TrainingMonitor(cfg.FIG_PATH, jsonPath=cfg.JSON_PATH,
                        startAt=args["start_epoch"]),
        EpochCheckpoint(args["checkpoints"], every=5,
                        startAt=args["start_epoch"])
    ]

    model.fit(trainGenerator.generator(), steps_per_epoch=trainGenerator.numImages // batchSize, validation_data=valGenerator.generator(),
              validation_steps=valGenerator.numImages // batchSize, epochs=10, max_queue_size=batchSize * 2, callbacks=callbacks, verbose=1)
    
    trainGenerator.close()
    valGenerator.close()


if "__main__" == __name__:
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-c", "--checkpoints", required=True,
                    help="path to output checkpoint directory")
    ap.add_argument("-m", "--model", type=str,
                    help="path to *specific* model checkpoint to load")
    ap.add_argument("-s", "--start_epoch", type=int, default=0,
                    help="epoch to restart training at")
    args = vars(ap.parse_args())
    main()
