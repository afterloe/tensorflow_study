#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from keras.applications.vgg16 import VGG16


def main():
    print("[info] loading network")
    model = VGG16(weights="imagenet", include_top=args["include_top"] > 0)
    print("[info] show layers")
    for (i, layer) in enumerate(model.layers):
        print("[info] {} \t {}".format(i, layer.__class__.__name__))


if "__main__" == __name__:
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-i", "--include_top", type=int, default=1, help="whether or not to include top of cnn")
    args = vars(ap.parse_args())
    main()
