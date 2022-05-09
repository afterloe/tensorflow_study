#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten, Dense
from keras import backend as K


class ShallowNet:
    @staticmethod
    def build(width: int, height: int, depth: int, classes: int):
        model = Sequential()
        inputShape = (height, width, depth)
        if "channels_first" == K.image_data_format():
            inputShape = (depth, height, width)
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
