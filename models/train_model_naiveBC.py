#!/usr/bin/env python3
import pandas as pd
import numpy as np
from os import listdir
from os.path import exists
from typing import Tuple
import tensorflow as tf
from helpers import data_and_label, generate_trajectory
from tensorflow.keras import layers, optimizers, losses, models
TRAIN="processed_data/train_datasets/train-start-c088196696e9f167c879.csv"
TEST="processed_data/train_datasets/test-start-c088196696e9f167c879.csv"
TRAIN="processed_data/train_datasets/train-start-time-5e9156387f59cb9efb35.csv"
TEST="processed_data/train_datasets/test-start-time-5e9156387f59cb9efb35.csv"
H_NEURONS=128
EPOCHS=20
OFILE=f"models/naiveBCx{H_NEURONS}x2-ep{EPOCHS}-norm-start-timesignal"
OVERWRITE=False
STARTINDEX=0
ID=""


def generate_trajectories_with_target(model, means: np.ndarray, deviations: np.ndarray, num=50, length=50):
    trajectories = []
    means = np.repeat(means.reshape(1,-1), num, axis=0)
    deviations = np.repeat(deviations.reshape(1,-1), num, axis=0)
    target_coords = np.random.normal(means, deviations, means.shape)
    initial_states = np.concatenate([np.zeros((num,9)), target_coords], axis=1)
    for init in initial_states:
        trajectories.append(generate_trajectory(model, init, length))
    return np.concatenate(trajectories, axis=0)
    
def generate_trajectories_with_start(model, means: np.ndarray, deviations: np.ndarray, num=50, length=50):
    trajectories = []
    means = np.repeat(means.reshape(1,-1), num, axis=0)
    deviations = np.repeat(deviations.reshape(1,-1), num, axis=0)
    start_params = np.random.normal(means, deviations, means.shape)
    initial_states = np.concatenate([np.zeros((num,1)), start_params, np.zeros((num,4))], axis=1)
    for init in initial_states:
        trajectories.append(generate_trajectory(model, init, length))
    return np.concatenate(trajectories, axis=0)



if __name__ == "__main__":
    np.random.seed(133)
    train_data, train_labels = data_and_label(pd.read_csv(TRAIN))
    test_data, test_labels = data_and_label(pd.read_csv(TEST))
    target_means = np.mean(train_data, axis=0)[-3:]
    target_sds = np.std(train_data, axis=0)[-3:]

    start_means = np.mean(train_data, axis=0)[:-4]
    start_means = start_means[1:] # discard time
    start_sds = np.std(train_data, axis=0)[:-4]
    start_sds = start_sds[1:] # discard time
   
    if not exists(OFILE) or OVERWRITE:
        model = tf.keras.Sequential([
            layers.Input(shape=train_data[0].shape),
            layers.Dense(H_NEURONS),
            layers.ReLU(),
            layers.Dense(H_NEURONS),
            layers.ReLU(),
            layers.Dense(train_labels[0].size, activation=None),
        ])

        model.compile(
            optimizer=optimizers.Adam(10e-5),
            loss=losses.Huber()
        ) 

        stop_condition = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5)

        history = model.fit(
            train_data,
            train_labels,
            epochs=EPOCHS,
            validation_data=(test_data, test_labels),
            validation_freq=1,
            verbose=2,
            callbacks=[stop_condition]
        )

        model.save(OFILE)
    else:
        model = models.load_model(OFILE)

    #start = pd.read_csv(TRAIN).values[STARTINDEX,:-8]
    #trajectory = generate_trajectory(model, start, 100)
    trajectory = generate_trajectories_with_target(model, target_means, target_sds)
    #trajectory = generate_trajectories_with_start(model, start_means, start_sds)
    cols = ["Time","x","y","z","rx", "ry", "rz", "rw", "Released", "xt", "yt", "zt"]
    df = pd.DataFrame(data=trajectory, columns=cols)
    #t = pd.Series(data=np.arange(0,5,0.01), name="Time")
    #output = pd.concat([t,df], axis=1)
    df.to_csv(OFILE+f"-{STARTINDEX}-{ID}.csv", index=False)
    pass