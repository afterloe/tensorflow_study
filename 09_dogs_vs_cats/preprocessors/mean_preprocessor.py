#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from numpy import ndarray
from cv2 import split, merge


class MeanPreprocessor:

    R: float = None
    G: float = None
    B: float = None

    def __init__(self, r: float, g: float, b: float):
        """
        均值处理器

        :param r: mean color r
        :param g: mean color g
        :param b: mean color b
        """
        self.R = r
        self.G = g
        self.B = b

    def preprocess(self, image: ndarray) -> ndarray:
        """
        处理器

        :param image: cv读取的图像
        :return:  处理之后的图像
        """
        b, g, r = split(image.astype("float32"))
        b -= self.B
        g -= self.G
        r -= self.R
        return merge([b, g, r])
