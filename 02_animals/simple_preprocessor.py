#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from cv2 import resize, INTER_AREA
from header import Preprocessor
from numpy import ndarray


class SimplePreprocessor(Preprocessor):

    def __init__(self, width, height, inter=INTER_AREA):
        """
        图像处理器构造方法

        :param width: 调整大小后输入图像的目标宽度
        :param height: 调整大小后输入图像的目标高度
        :param inter: 一个可选参数，用于控制调整大小时使用的插值算法
        """
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image: ndarray) -> ndarray:
        return resize(image, (self.width, self.height), interpolation=self.inter)
