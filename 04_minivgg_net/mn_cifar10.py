#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from numpy import floor
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.python.keras.optimizers import gradient_descent_v2
from keras.datasets import cifar10
from tensorflow.python.keras.callbacks import LearningRateScheduler, ModelCheckpoint

from minivgg_network import MiniVGGNet


def show_in_plt(H, epochs: int = 100):
    import matplotlib.pyplot as plt
    from numpy import arange
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(arange(0, epochs), H.history["accuracy"], label="accuracy")
    plt.plot(arange(0, epochs), H.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


def step_decay(epoch: int) -> float:
    initAlpha = 0.01
    factor = 0.25
    dropEvery = 5
    alpha = initAlpha * (factor ** floor((1 + epoch) / dropEvery))

    return float(alpha)


def main():
    print("[info] load database")
    dataset = cifar10.load_data()
    (trainX, trainY), (testX, testY) = dataset
    trainX = trainX.astype("float") / 255.0
    testX = testX.astype("float") / 255.0

    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)
    labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    print("[info] compiling model ...")
    opt = gradient_descent_v2.SGD(learning_rate=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("[info] training network")
    # checkpoint = ModelCheckpoint(r"D:/Datasets/output/best_weights.hdf5", monitor="val_loss", save_best_only=True,
    #                              verbose=1)
    callbacks = [LearningRateScheduler(step_decay)]
    # callbacks = [checkpoint]
    H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40,
                  callbacks=callbacks, verbose=1)

    print("[info] evaluating network ...")
    predictions = model.predict(testX, batch_size=64)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))
    show_in_plt(H, epochs=40)


if "__main__" == __name__:
    main()
