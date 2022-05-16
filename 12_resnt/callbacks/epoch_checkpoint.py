#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from keras.callbacks import Callback
from os import sep


class EpochCheckpoint(Callback):

    filePath: str = None
    every: int = None
    epochNum: int = None

    def __init__(self, filePath: str, every: int = 5, startAt: int = 0):
        super(Callback, self).__init__()
        self.filePath = filePath
        self.every = every
        self.epochNum = startAt

    def on_epoch_end(self, epoch, logs={}):
        if (self.epochNum + 1) % self.every == 0:
            print("[info] save checkpoin")
            p = sep.join(
                [self.filePath, "epoch_{}.hdf5".format(self.epochNum + 1)])
            self.model.save(p, overwrite=True)
        self.epochNum += 1
