#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from keras.models import load_model
from numpy import array
from json import loads
from progressbar import Percentage, Bar, ETA, ProgressBar

from config import config as cfg
from preprocessors import SimplePreprocessor, MeanPreprocessor, ImageToArrayPreprocessor, CropPreprocessor
from hdf5io import HDF5DatasetGenerator
from utils.ranked import rank5_accuracy

means = loads(open(cfg.DATASET_MEAN, "r").read())

sp = SimplePreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
cp = CropPreprocessor(227, 227)
iap = ImageToArrayPreprocessor()

print("[info] loading model")
model = load_model(cfg.MODEL_PATH)

batchSize: int = 64

print("[info] predicting on test data (no crops)")
testGenerator = HDF5DatasetGenerator(cfg.TEST_HDF5, batchSize, preprocessors=[sp, mp, iap], classes=2)
predictions = model.predict(testGenerator.generator(), steps=testGenerator.numImages // batchSize,
                                      max_queue_size=batchSize * 2)

rank1, _ = rank5_accuracy(predictions, testGenerator.db["labels"])
print("[info] rank-1: %.2f%%" % (rank1 * 100))

testGenerator = HDF5DatasetGenerator(cfg.TEST_HDF5, batchSize, preprocessors=[mp], classes=2)
predictions = []

widgets = ["Evaluating: ", Percentage(), " ", Bar(), " ", ETA()]
pbar = ProgressBar(maxval=testGenerator.numImages // batchSize, widgets=widgets)
pbar.start()

for (i, (images, labels)) in enumerate(testGenerator.generator(passes=1)):
    for image in images:
        crops = cp.preprocess(image)
        crops = array([iap.preprocess(c) for c in crops], dtype="float32")
        prediction = model.predict(crops)
        predictions.append(prediction.mean(axis=0))
    pbar.update(i)
pbar.finish()

print("[info] predicting on test data (with crops) ...")
rank1, _ = rank5_accuracy(predictions, testGenerator.db["labels"])
print("[info] rank-1: %.2f%%" % (rank1 * 100))
testGenerator.close()
