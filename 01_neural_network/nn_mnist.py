#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets


def main():
    print("begin load MNIST (sample) dataset ... ...")
    digits = datasets.load_digits()
    data = digits.data.astype("float")
    data = (data - data.min()) / (data.max() - data.min())
    print("[info] samples: {}, dim: {}".format(data.shape[0], data.shape[1]))
    trainX, testX, trainY, testY = train_test_split(data, digits.target, test_size=0.25)
    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)
    print("[info] training network ... ...")
    network = NeuralNetwork([trainX.shape[1], 32, 16, 10])
    print("[info] {}".format(network))
    network.fit(trainX, trainY, epochs=1000)
    print("[info] evaluating network ... ...")
    predictions = network.predict(testX)
    predictions = predictions.argmax(axis=1)
    print(classification_report(testY.argmax(axis=1), predictions))


if "__main__" == __name__:
    from neural_network import NeuralNetwork
    main()
