#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from cv2 import INTER_AREA, resize, flip
from numpy import ndarray, array


class CropPreprocessor:

    width: int = None
    height: int = None
    horizontal: bool = None
    inter: int = None

    def __init__(self, width: int, height: int, horizontal: bool = True, inter: int = INTER_AREA):
        self.width = width
        self.height = height
        self.horizontal = horizontal
        self.inter = inter

    def preprocess(self, image: ndarray) -> array:
        crops = []
        h, w = image.shape[:2]
        coordinates = [
            [0, 0, self.width, self.height],
            [w - self.width, 0, w, self.height],
            [w - self.width, h - self.height, w, h],
            [0, h - self.height, self.width, h]
        ]
        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (h - self.height))
        coordinates.append([dW, dH, w - dW, h - dH])

        for (startX, startY, endX, endY) in coordinates:
            crop = image[startY: endY, startX: endX]
            crop = resize(crop, (self.width, self.height), interpolation=self.inter)
            crops.append(crop)

        if self.horizontal:
            mirrors = [flip(c, 1) for c in crops]
            crops.extend(mirrors)

        return array(crops)
