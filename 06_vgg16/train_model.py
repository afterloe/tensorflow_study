#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from h5py import File
from numpy import unique
from pickle import dumps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def main():
    db = File(args["db"], "r")
    labels = [str(x).replace("b'", "") for x in unique(db["label_names"])]

    i = int(db["labels"].shape[0] * 0.75)
    print("[info] tuning hyperparameters ...")
    params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
    model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=args["jobs"])
    model.fit(db["features"][:i], db["labels"][:i])
    print("[info] best hyperparameters: {}".format(model.best_params_))

    print("[info] evaluating ...")
    predictions = model.predict(db["features"][i:])
    print("[info] network report")
    print(classification_report(db["labels"][i:], predictions, target_names=labels))

    print("[info] saving model ...")
    f = open(args["model"], "wb")
    f.write(dumps(model.best_estimator_))
    f.close()

    db.close()


if "__main__" == __name__:
    r"""
    python train_model.py -d D:\Datasets\output\features.hdf5 -m D:\Datasets\output\animals.cpickle
    """
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-d", "--db", required=True, help="path HDF5 database")
    ap.add_argument("-m", "--model", required=True, help="path to output model")
    ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs to run when tuning hyperparameters")
    args = vars(ap.parse_args())
    main()
