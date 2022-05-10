#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from tensorflow.python.keras.optimizers import gradient_descent_v2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from tensorflow.python.keras import backend as T

from le_network import LeNet, show_in_plt


def main():
    print("[info] accessing MNIST ...")
    # dataset = datasets.fetch_openml("mnist_784")
    dataset = datasets.fetch_openml("mnist_784", data_home=r"D:/Datasets/sklearn")
    data = dataset.data
    if T.image_data_format() == "channels.first":
        data = data.values.reshape(data.shape[0], 1, 28, 28)
    else:
        data = data.values.reshape(data.shape[0], 28, 28, 1)
    trainX, testX, trainY, testY = train_test_split(data / 255.0, dataset.target.astype("int"), test_size=0.25,
                                                    random_state=42)
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)
    print("[info] compiling model ...")
    opt = gradient_descent_v2.SGD(learning_rate=0.01)
    model = LeNet.build(28, 28, 1, 10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=20, verbose=1)
    print("[info] evaluating network ...")
    predictions = model.predict(testX, batch_size=128)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                                target_names=[str(item) for item in lb.classes_]))
    show_in_plt(H, 20)


if "__main__" == __name__:
    main()
