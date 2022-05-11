#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from cv2 import INTER_AREA, resize
from numpy import ndarray


class AspectAwarePreprocessor:
    width: int = None
    height: int = None
    inter: int = None

    def __init__(self, width: int, height: int, inter: int = INTER_AREA):
        """
        初始化

        :param width: 图像预期宽度
        :param height:  图像预期高度
        :param inter:  插值算法
        """
        self.width = width
        self.height = height
        self.inter = inter

    @staticmethod
    def rs(image, width=None, height=None, inter=INTER_AREA):
        dim: int = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)

        else:
            r = width / float(w)
            dim = (width, int(h * r))

        resized = resize(image, dim, interpolation=inter)

        return resized

    def preprocess(self, image: ndarray):
        height, width = image.shape[:2]
        dW = 0
        dH = 0
        if width < height:
            image = AspectAwarePreprocessor.rs(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)
        else:
            image = AspectAwarePreprocessor.rs(image, height=self.height, inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)
        h, w = image.shape[:2]
        image = image[dH: h - dH, dW:w - dW]
        return resize(image, (self.width, self.height), interpolation=self.inter)
