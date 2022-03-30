#!/usr/bin/env python3
import pandas as pd
import numpy as np
from os import listdir
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, activations, regularizers, models
TRAIN="processed_data/train_datasets/train-003430811ff20c35ccd5.csv"
TEST="processed_data/train_datasets/test-003430811ff20c35ccd5.csv"

def data_and_label(df: pd.DataFrame) -> Tuple[np.ndarray]:
    data_cols = df.columns[:-8]
    label_cols = df.columns[-8:]
    return df[data_cols].values, df[label_cols].values

def generate_trajectory(model, initial_state):
    pass


if __name__ == "__main__":
    train_data, train_labels = data_and_label(pd.read_csv(TRAIN))
    test_data, test_labels = data_and_label(pd.read_csv(TEST))
   
    model = tf.keras.Sequential([
        layers.Dense(512),
        layers.ReLU(),
        layers.Dense(512),
        layers.ReLU(),
        layers.Dense(train_labels[0].size, activation=None),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss=losses.Huber()
    )

    history = model.fit(
        train_data,
        train_labels,
        batch_size=4096,
        epochs=20,
        validation_data=(test_data, test_labels),
        validation_freq=1,
        verbose=2,
    )

    pass
