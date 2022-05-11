#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from numpy import ndarray
from sklearn.feature_extraction.image import extract_patches_2d


class PatchPreprocessor:

    width: int = None
    height: int = None

    def __init__(self, width: int, height: int):
        """
        随机图像剪裁器

        :param width: 目标图像宽度
        :param height:  目标图像高度
        """
        self.width = width
        self.height = height

    def preprocess(self, image: ndarray) -> ndarray:
        """
        图像处理

        :param image:cv读取的图像
        :return: 处理之后的图像
        """
        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]
