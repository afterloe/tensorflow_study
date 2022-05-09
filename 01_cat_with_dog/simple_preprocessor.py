#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from cv2 import resize, INTER_AREA
from numpy import ndarray


class SimplePreprocessor:
    """
    分类处理前需要使用图像处理器对图像进行统一处理，避免大小等图像因素干扰分类
    """

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
        """
        处理逻辑

        :param image: cv读取的图像
        :return:ndarray: 缩放后的图像
        """
        return resize(image, (self.width, self.height), interpolation=self.inter)
