#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from numpy import expand_dims


def main():
    image = load_img(args["image"])
    image = img_to_array(image)
    image = expand_dims(image, axis=0)
    """
        rotation_range 参数控制随机旋转的度数范围。允许输入图像随机旋转±30度。
        width_shift_range 宽度偏移范围用于水平偏移。参数值是给定维度的一小部分，在本例中为10%。
        height_shift_range 高度偏移范围用于垂直偏移。参数值是给定维度的一小部分，在本例中为10%。
        shear_range 剪切范围控制逆时针方向的角度，即允许剪切图像的弧度。
        zoom_range（缩放范围），这是一个浮点值，允许根据以下均匀分布的值对图像进行“放大”或“缩小”：[1-zoom_range，1+zoom_range]。
        horizontal_flip 水平翻转布尔值控制在训练过程中是否允许水平翻转给定的输入。
    """
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                             zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
    total = 0
    print("[info] generating image")
    imageGenerator = aug.flow(image, batch_size=1, save_to_dir=args["out"], save_prefix=args["prefix"],
                              save_format="jpg")
    for _ in imageGenerator:
        total += 1
        if total == 10:
            break
    print("[info] image generator success in %s " % args["out"])


if "__main__" == __name__:
    """
    python augmentation_demo.py -i D:\Datasets\animals\sample\00000043.jpg -o ./
    
    随机生成图像素材
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the input image")
    ap.add_argument("-o", "--out", required=True, help="path to output directory to store augmentation example")
    ap.add_argument("-p", "--prefix", type=str, default="image", help="output filename prefix")
    args = vars(ap.parse_args())
    main()
