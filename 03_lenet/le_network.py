#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.core import Activation, Flatten, Dense
from tensorflow.python.keras import backend as T


class LeNet:

    @staticmethod
    def build(width: int, height: int, depth: int, classes: int):
        model = Sequential()
        inputShape = (height, width, depth)
        if T.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(20, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


def show_in_plt(H, epochs: int = 100):
    import matplotlib.pyplot as plt
    from numpy import arange
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(arange(0, epochs), H.history["accuracy"], label="accuracy")
    plt.plot(arange(0, epochs), H.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()
