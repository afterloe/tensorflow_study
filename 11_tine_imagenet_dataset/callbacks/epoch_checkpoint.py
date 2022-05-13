#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from keras.callbacks import ModelCheckpoint


class EpochCheckpoint(ModelCheckpoint):

    filePath: str = None
    every: int = None
    startAt: int = None

    def __init__(self, filePath: str, every: int = 5, startAt: int = 0):
        super(ModelCheckpoint, self).__init__(
            filePath, monitor="val_loss", mode="min", verbose=1)
        self.filePath = filePath
        self.every = every
        self.startAt = startAt

    def on_epoch_end(self, epoch, logs=None):
        if 0 == epoch // self.every:
            return super().on_epoch_end(epoch, logs)
