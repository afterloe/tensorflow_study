#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from keras.preprocessing.image import img_to_array
from numpy import ndarray


class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    def preprocess(self, image: ndarray) -> ndarray:
        return img_to_array(image, data_format=self.dataFormat)
