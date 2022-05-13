#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from h5py import File
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from pickle import dumps


def main():
    dataset = File(args["dataset"], "r")
    idx = int(dataset["labels"].shape[0] * 0.75)
    print("[info] tuning hyper parameters")
    params = {"C": [0.001, 0.01, 0.1, 1.0]}
    model = GridSearchCV(LogisticRegression(), params,
                         cv=3, n_jobs=args["jobs"])
    print("[info] fit model")
    model.fit(dataset["features"][:idx], dataset["labels"][:idx])
    print("[info] best hyper parameter is %f", model.best_params_)

    print("[info] evaluating")
    predicts = model.predict(dataset["feature"][idx:])
    print(classification_report(
        dataset["labels"][idx:], predicts, target_names=dataset["label_names"]))
    acc = accuracy_score(dataset["labels"][idx:], predicts)
    print("[info] score: %f%%" % (acc * 100))

    print("[info] saving model")
    f = open(args["model"], "wb")
    f.write(dumps(model.best_estimator_))
    f.close()

    dataset.close()


if "__main__" == __name__:
    r"""
        python train_model.py -d D:\Datasets\output\kaggle_dogs_vs_cats\hdf5\features.hdf5 -m D:\Datasets\output\kaggle_dogs_vs_cats\dogs_vs_cats.pickle
    """
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path HDF5 dataset")
    ap.add_argument("-m", "--model", required=True,
                    help="path to output model")
    ap.add_argument("-j", "--jobs", type=int, default=-1,
                    help="# of jobs to run when tuning hyper parameters")
    args = vars(ap.parse_args())
    main()
