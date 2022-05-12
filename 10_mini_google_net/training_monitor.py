#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import matplotlib.pyplot as plt

from keras.callbacks import BaseLogger
from os.path import exists
from json import loads, dumps
from numpy import arange


class TrainingMonitor(BaseLogger):

    figPath: str = None
    jsonPath: str = None
    startAt: int = 0
    H: dict = None

    def __init__(self, figPath: str, jsonPath: str = None, startAt: int = 0):
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs={}):
        self.H = {}
        if self.jsonPath is not None:
            if exists(self.jsonPath):
                self.H = loads(open(self.jsonPath).read())
                if self.startAt > 0:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        for (k, v) in logs.items():
            log = self.H.get(k, [])
            log.append(v)
            self.H[k] = log
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(dumps(self.H))
            f.close()
        if len(self.H["loss"]) > 1:
            N = arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["accuracy"], label="accuracy")
            plt.plot(N, self.H["val_accuracy"], label="val_accuracy")
            plt.title(
                "Training Loss and Accuracy [Epoch %d]" % len(self.H["loss"]))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            plt.savefig(self.figPath)
            plt.close()
