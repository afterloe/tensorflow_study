#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from h5py import File, Dataset, special_dtype
from os.path import exists


class HDF5DatasetWriter:

    db: File = None
    data: Dataset = None
    labels: Dataset = None
    buffer: dict = None
    bufferSize: int = 1000
    idx: int = 0

    def __init__(self, dims: tuple, outputPath: str, dataKey: str = "images", buffSize: int = 1000):
        if exists(outputPath):
            raise ValueError("The supplied 'outputPath' already exists and cannot be overwritten.")
        self.db = File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")
        self.bufferSize = buffSize
        self.buffer = {"data": [], "labels": []}

    def add(self, rows, labels):
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        if len(self.buffer["data"]) >= self.bufferSize:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def close(self):
        if len(self.buffer["data"]) > 0:
            self.flush()
        self.db.close()

    def storeClassLabels(self, classLabels):
        dt = special_dtype(vlen=str)
        labelSet = self.db.create_dataset("label_names", (len(classLabels), ), dtype=dt)
        labelSet[:] = classLabels
