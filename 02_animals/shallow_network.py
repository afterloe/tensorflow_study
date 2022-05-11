#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten, Dense
from keras import backend as K


def show_in_plt(H, value: int = 100):
    import matplotlib.pyplot as plt
    from numpy import arange
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(arange(0, value), H.history["loss"], label="train_loss")
    plt.plot(arange(0, value), H.history["val_loss"], label="val_loss")
    plt.plot(arange(0, value), H.history["accuracy"], label="accuracy")
    plt.plot(arange(0, value), H.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


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
