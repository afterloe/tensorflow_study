#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


class NeuralNetwork:
    """
        Neural Network - 神经网络基础
    """
    def __init__(self, layers: list, alpha: float = 0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha
        for i in np.arange(0, len(layers) - 2):
            w = np.random.rand(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))
        w = np.random.rand(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        return "NeuralNetwork: {}".format("-".join(str(layer) for layer in self.layers))

    def fit(self, X: np.ndarray, y, epochs=1000, displayUpdate=100):
        X = np.c_[X, np.ones((X.shape[0]))]
        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[info] epoch=%d, loss=%.7f" % (epoch + 1, loss))

    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)]
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer])
            out = sigmoid(net)
            A.append(out)
        error = A[-1] - y
        D = [error * sigmoid_deriv(A[-1])]
        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * sigmoid_deriv(A[layer])
            D.append(delta)
        D = D[::-1]
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias: bool = True):
        p = np.atleast_2d(X)
        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]
        for layer in np.arange(0, len(self.W)):
            p = sigmoid(np.dot(p, self.W[layer]))
        return p

    def calculate_loss(self, X, target):
        targets = np.atleast_2d(target)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        return loss
