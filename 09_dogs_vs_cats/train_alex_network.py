#!/usr/bin/env python3
# -*- coding=utf-8 -*-


from os import sep, getpid

import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam_v2

from json import loads
from config import config as cfg
from preprocessors import SimplePreprocessor, PatchPreprocessor, MeanPreprocessor, ImageToArrayPreprocessor
from hdf5io import HDF5DatasetGenerator
from nn.conv.alex_network import AlexNet
from callbacks.training_monitor import TrainingMonitor

matplotlib.use("agg")
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
means = loads(open(cfg.DATASET_MEAN).read())

sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

trainGenerator = HDF5DatasetGenerator(cfg.TRAIN_HDF5, 128, aug=aug, classes=2, preprocessors=[pp, mp, iap])
valGenerator = HDF5DatasetGenerator(cfg.VAL_HDF5, 128, classes=2, preprocessors=[sp, mp, iap])

print("[info] compiling model")
opt = adam_v2.Adam(learning_rate=1e-3)
model = AlexNet.build(width=227, height=227, depth=3, classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

pngOutputPath = sep.join([cfg.OUTPUT_PATH, "%s.png" % getpid()])
callbacks = [TrainingMonitor(pngOutputPath)]

model.fit_generator(
    trainGenerator.generator(),
    steps_per_epoch=trainGenerator.numImages // 128,
    validation_data=valGenerator.generator(),
    validation_steps=valGenerator.numImages // 128,
    epochs=75,
    max_queue_size=128 * 2,
    callbacks=callbacks,
    verbose=1
)

print("[info] serializing model")
model.save(cfg.MODEL_PATH, overwrite=True)

trainGenerator.close()
valGenerator.close()
