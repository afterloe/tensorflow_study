#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from h5py import File
from keras.utils import np_utils
from numpy import arange, array, inf


class HDF5DatasetGenerator:

    db: File = None
    numImages: int = None
    batchSize: int = None
    preprocessors: list = None
    aug = None
    binarize: bool = None
    classes: int = None

    def __init__(self, dbPath: str, batchSize: int, preprocessors: list = None, aug=None, binarize: bool = True,
                 classes: int = 2):
        """
        数据生成器

        :param dbPath: hdf5数据集位置
        :param batchSize:  数据批次
        :param preprocessors: 数据预处理器
        :param aug: 数据增强器
        :param binarize: 标签是否进行二进制化处理
        :param classes: 分类数量
        """
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

        self.db = File(dbPath)
        self.numImages = self.db["labels"].shape[0]

    def generator(self, passes: int = inf):
        """
        数据生成器

        :param passes: 循环次数上限，默认为无限
        :return:
        """
        epochs: int = 0
        while epochs < passes:
            for i in arange(0, self.numImages, self.batchSize):
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]
                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.classes)
                if self.preprocessors is not None:
                    procImages = []
                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                        procImages.append(image)
                    images = array(procImages)
                if self.aug is not None:
                    images, labels = next(self.aug.flow(images, labels, batch_size=self.batchSize))
                yield images, labels
            epochs += 1

    def close(self):
        self.db.close()
        pass
