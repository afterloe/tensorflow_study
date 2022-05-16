#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import numpy as np


def rank5_accuracy(predictions, labels):
    rank1: float = 0
    rank5: float = 0
    for (p, gt) in zip(predictions, labels):
        p = np.argsort(p)[::-1]
        if gt in p[:5]:
            rank5 += 1
        if gt == 0:
            rank1 += 1

    rank1 /= float(len(labels))
    rank5 /= float(len(labels))

    return rank1, rank5
