#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import tensorflow as tf

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.core import Activation, Flatten, Dropout, Dense
from tensorflow.python.keras import backend as T


class MiniVGGNet:

    @staticmethod
    def build(width: int, height: int, depth: int, classes: int):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        if "channels_first" == T.image_data_format():
            inputShape = (depth, height, width)
            chanDim = 1
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        # model.add(BatchNormalization())
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
