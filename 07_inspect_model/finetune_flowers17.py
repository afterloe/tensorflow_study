#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.models import Model
from keras.optimizers import gradient_descent_v2
from imutils.paths import list_images
from os import sep
from numpy import unique

from aspect_aware_preprocessor import AspectAwarePreprocessor
from image_to_array_preprocessor import ImageToArrayPreprocessor
from simple_dataset_loader import SimpleDatasetLoader
from fchead_network import FCHeadNet


def main():
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                             zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
    print("[info] loading image ...")
    imagePaths = list(list_images(args["dataset"]))
    """
    eg:
        dataset_name/{class_name}/example.jpg
    """
    classNames = [p.split(sep)[-2] for p in imagePaths]
    classNames = [str(x) for x in unique(classNames)]
    dataset = SimpleDatasetLoader(preprocessors=[AspectAwarePreprocessor(224, 224), ImageToArrayPreprocessor()])
    data, labels = dataset.load(imagePaths, verbose=500)
    data = data.astype("float") / 255.0
    trainX, testX, trainY, testY = train_test_split(data, labels, train_size=0.25, random_state=42)
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)
    baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    headModel = FCHeadNet.build(baseModel, len(classNames), 256)
    model = Model(inputs=baseModel.input, outputs=headModel)
    for layer in baseModel.layers:
        layer.trainable = False
    print("[info] compiling model ...")
    opt = gradient_descent_v2.SGD(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    print("[info] training head ...")
    model.fit(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), epochs=25,
              steps_per_epoch=len(trainX) // 32, verbose=1)
    print("[info] evaluating after initialization ...")
    predictions = model.predict(testX, batch_size=32)
    print("[info] network report")
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

    for layer in baseModel.layers[15:]:
        layer.trainable = True

    print("[info] re-compiling model ...")
    opt = gradient_descent_v2.SGD(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    print("[info] fine-tuning model ...")
    model.fit(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), epochs=100,
              steps_per_epoch=len(trainX) // 32, verbose=1)
    print("[info] evaluating after fine-tuning ...")
    predictions = model.predict(testX, batch_size=32)
    print("[info] network report")
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

    print("[info] serializable model ...")
    model.save(args["model"])


if "__main__" == __name__:
    r"""
        python finetune_flowers17.py -d D:\Datasets\Flowers-17 -m D:\Datasets\output\flowers17.model
    """
    from argparse import ArgumentParser

    ap = ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    ap.add_argument("-m", "--model", required=True, help="path ot output model")
    args = vars(ap.parse_args())
    main()
