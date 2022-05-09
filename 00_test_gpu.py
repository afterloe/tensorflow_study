#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import tensorflow as tf


def main():
    device_name = tf.test.gpu_device_name()
    print("build with CUDA: %s" % tf.test.is_built_with_cuda())
    print("supper GPU: %s" % tf.test.is_gpu_available())
    print("GPU device: %s" % device_name)


if "__main__" == __name__:
    main()
