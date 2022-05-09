#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from abc import ABC, abstractmethod
from numpy import ndarray


class Preprocessor(ABC):
    """
    分类处理前需要使用图像处理器对图像进行统一处理，避免大小等图像因素干扰分类
    """

    width = 0
    height = 0

    @abstractmethod
    def preprocess(self, image: ndarray) -> ndarray:
        """
        处理逻辑

        :param image: 读取的图像
        :return:ndarray: 缩放后的图像
        """
        pass