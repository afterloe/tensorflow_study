#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import numpy as np


def main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    network = NeuralNetwork([2, 1], alpha=0.5)
    network.fit(X, y, epochs=20000)
    for (x, target) in zip(X, y):
        pred = network.predict(x)[0][0]
        step = 1 if pred > 0.5 else 0
        print("[info] data={}, ground-truth={}, pred={:.4f}, step={}".format(x, target[0], pred, step))


if "__main__" == __name__:
    from neural_network import NeuralNetwork
    main()
