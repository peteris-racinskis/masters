#!/usr/bin/env python3
import pandas as pd
import numpy as np
from os import listdir
from os.path import exists
from typing import Tuple
import tensorflow as tf
from helpers import data_and_label, generate_trajectory
from tensorflow.keras import layers, optimizers, losses, models
TRAIN="processed_data/train_datasets/train-003430811ff20c35ccd5.csv"
TEST="processed_data/train_datasets/test-003430811ff20c35ccd5.csv"
OFILE="models/naiveBC-small-movement-thresh"
OVERWRITE=False
STARTINDEX=156


if __name__ == "__main__":
    np.random.seed(133)
    train_data, train_labels = data_and_label(pd.read_csv(TRAIN))
    test_data, test_labels = data_and_label(pd.read_csv(TEST))
   
    if not exists(OFILE) or OVERWRITE:
        model = tf.keras.Sequential([
            layers.Input(shape=train_data[0].shape),
            layers.Dense(128),
            layers.ReLU(),
            layers.Dense(128),
            layers.ReLU(),
            layers.Dense(train_labels[0].size, activation=None),
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=losses.Huber()
        )

        history = model.fit(
            train_data,
            train_labels,
            epochs=4,
            validation_data=(test_data, test_labels),
            validation_freq=1,
            verbose=2,
        )

        model.save(OFILE)
    else:
        model = models.load_model(OFILE)

    start = pd.read_csv(TRAIN).values[STARTINDEX,:-8]
    trajectory = generate_trajectory(model, start, 100)
    cols = ["x","y","z","rx", "ry", "rz", "rw", "Released", "xt", "yt", "zt"]
    df = pd.DataFrame(data=trajectory, columns=cols)
    t = pd.Series(data=np.arange(0,5,0.01), name="Time")
    output = pd.concat([t,df], axis=1)
    output.to_csv(OFILE+f"-{STARTINDEX}-movement-thresh.csv", index=False)
    pass