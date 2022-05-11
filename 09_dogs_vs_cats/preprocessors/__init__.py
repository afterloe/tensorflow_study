#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from abc import ABC, abstractmethod
from numpy import ndarray, array


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


class DatasetLoader(ABC):
    """
    数据加载器： 实现数据集加载
    """
    preprocessors = None

    @abstractmethod
    def load(self, imagePaths: str, verbose: int = -1) -> (array, array):
        """
        图像加载

        :param imagePaths: 图像路径， 例如：/path/to/dataset/{class}/{image}.jpg
        :param verbose: 进度分类，用于将处理图像的进度更新打印到控制台
        :return: (array, array): 数据集 与 标签集
        """
        pass