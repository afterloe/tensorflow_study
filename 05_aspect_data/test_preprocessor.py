#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import cv2 as cv


def main():
    p = AspectAwarePreprocessor(64, 64)
    image = cv.imread(r"D:\Datasets\animals\sample\00000043.jpg", cv.IMREAD_COLOR)
    cv.imshow("src", image)
    image = p.preprocess(image)
    cv.imshow("dst", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if "__main__" == __name__:
    from aspect_aware_preprocessor import AspectAwarePreprocessor
    main()
