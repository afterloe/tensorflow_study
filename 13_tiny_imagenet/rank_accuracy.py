#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from json import loads
from keras.models import load_model

from config import tiny_imagenet_config as cfg
from preprocessors import ImageToArrayPreprocessor, SimplePreprocessor, MeanPreprocessor
from utils.ranked import rank5_accuracy
from hdf5io import HDF5DatasetGenerator

means = loads(open(cfg.DATASET_MEAN, "r").read())

sp =SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

batchSize: int = 64
testGenerator = HDF5DatasetGenerator(cfg.TEST_HDF5, batchSize, preprocessors=[sp, mp, iap], classes=cfg.NUM_CLASSES)

print("[info] loading model")
model = load_model(cfg.MODEL_PATH)

print("[info] predicting ont test data")
predictions = model.predict_generator(testGenerator.generator(), steps=testGenerator.numImages // batchSize, max_queue_size=batchSize * 2)

rank_1, rank_5 = rank5_accuracy(predictions, testGenerator.db["labels"])
print("[info] rank-1: %.2f%%" % (rank_1 * 100))
print("[info] rank-5: %.2f%%" % (rank_5 * 100))

testGenerator.close()
