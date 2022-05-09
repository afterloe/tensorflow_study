#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import os

from cv2 import imread, IMREAD_COLOR
from header import DatasetLoader, Preprocessor
from numpy import array


class SimpleDatasetLoader(DatasetLoader):
    """
    数据加载器： 实现数据集加载
    """
    def __init__(self, preprocessors: Preprocessor = None):
        """
        加载器初始化

        :param preprocessors:  加载器实现子类， 默认为 空
        """
        self.preprocessors = preprocessors
        if None is self.preprocessors:
            self.preprocessors = []

    def load(self, imagePaths: str, verbose: int = -1) -> (array, array):
        data = []
        labels = []
        for (i, imagePath) in enumerate(imagePaths):
            image = imread(imagePath, IMREAD_COLOR)
            label = imagePath.split(os.path.seq)[-2]
            if None is not self.preprocessors:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            data.append(image)
            labels.append(label)
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[info]: processed %d/%d" % (i + 1, len(imagePaths)))
        return array(data), array(labels)
