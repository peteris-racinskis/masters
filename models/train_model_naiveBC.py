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
TEST="processed_data/train_datasets/test-start-time-doubled-7db3d40f19abc9f24f46-prep.csv"
TRAIN="processed_data/train_datasets/train-start-time-doubled-7db3d40f19abc9f24f46-prep.csv"
H_NEURONS=256
EPOCHS=50
OFILE=f"models/naiveBCx{H_NEURONS}x2-newdata-fixed-data-fastrelease-ep{EPOCHS}-norm-start-timesignal"
OVERWRITE=False
STARTINDEX=0
ID=""
TIME="Time"
REPEATS_IN_DATASET=1
PREPEND=True
huber = losses.Huber()
entropy = losses.BinaryCrossentropy(from_logits=True)

def generate_trajectories_with_target(model, means: np.ndarray, deviations: np.ndarray, num=50, length=50):
    trajectories = []
    means = np.repeat(means.reshape(1,-1), num, axis=0)
    deviations = np.repeat(deviations.reshape(1,-1), num, axis=0)
    target_coords = np.random.normal(means, deviations, means.shape)
    #tc = np.asarray([-2.5049639 , -0.03949555, -0.30162135])
    #target_coords = np.repeat(tc.reshape(1,-1), num, axis=0)
    #init_orientation = np.asarray([-0.18197935000062, 0.750300034880638, 0.247823745757341, 0.578429348766804])
    init_orientation = np.asarray([-0.595393347740173,-0.037378676235676, 0.794532498717308,0.04247132204473])
    init_orientations = np.repeat(init_orientation.reshape(1,-1), num, axis=0)
    initial_states = np.concatenate([np.zeros((num,4)), init_orientations, np.zeros((num,1)), target_coords], axis=1)
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

def quaternion_normalized_huber_loss(y_true, y_pred):
    huber_term = huber(y_true, y_pred)
    quat_reg = tf.reduce_sum(tf.square(tf.reduce_sum(tf.square(y_pred[:,3:7]), axis=-1) - 1))
    return huber_term + 0.1 * quat_reg

def partitioned_loss(y_true, y_pred):
    huber_term = huber(y_true[:,:3], y_pred[:,:3])
    #position_term = tf.reduce_sum(tf.square(y_true[:,:3] - y_pred[:,:3]))
    orientation_term = tf.reduce_sum(tf.math.reduce_euclidean_norm(y_true[:,3:7] - y_pred[:,3:7], axis=-1))
    release_term = entropy(y_true[:,-1], y_pred[:,-1])
    return huber_term + 0.1 * orientation_term + 0.1 * release_term

def quaternion_norm(df: pd.DataFrame):
    quat_cols = ["r"+c for c in "xyzw"]
    quat_norm_series = df[quat_cols].pow(2).sum(axis=1).pow(1/2)
    quat_norm_series.name = "quaternion_norm"
    return pd.concat([df,quat_norm_series], axis=1), True

def validation_on_test(df: pd.DataFrame, model, prep=PREPEND):
    stop_offset = -1 * REPEATS_IN_DATASET if prep else 0
    start_indices = list(df.loc[lambda d: d[TIME] == 0].index)
    start_indices += [df.index[-1] - stop_offset + 2]
    demos_with_trajectories = []
    columns = list(df.columns[:12])
    columns += [c for c in "xyz"]
    columns += ["r"+c for c in "xyzw"]
    columns += ["Released-model"]
    for i, ii in zip(start_indices[:-1], start_indices[1:]):
        demo, _ = data_and_label(df.iloc[i:ii+stop_offset], shuffle=False)
        initial_state = demo[0:1]
        trajectory = generate_trajectory(model, initial_state, len(demo))
        demos_with_trajectories.append(np.concatenate([demo, trajectory[:,1:-3]], axis=1))
    combined_data = np.concatenate(demos_with_trajectories, axis=0)
    combined_df = pd.DataFrame(data=combined_data, columns=columns)
    return combined_df

if __name__ == "__main__":
    np.random.seed(133)
    train_data, train_labels = data_and_label(pd.read_csv(TRAIN))
    test_data, test_labels = data_and_label(pd.read_csv(TEST))
    partitioned_loss(train_labels[0:10], train_labels[1:11])
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
            loss=huber
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
            #callbacks=[stop_condition]
        )

        model.save(OFILE)
    else:
        custom_objects = {
            "quaternion_normalized_huber_loss": quaternion_normalized_huber_loss,
            "partitioned_loss": partitioned_loss
        }
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = models.load_model(OFILE)

    #start = pd.read_csv(TRAIN).values[STARTINDEX,:-8]
    #trajectory = generate_trajectory(model, start, 100)
    trajectory = generate_trajectories_with_target(model, target_means, target_sds)
    validated_dataset = validation_on_test(pd.read_csv(TEST), model)
    #trajectory = generate_trajectories_with_start(model, start_means, start_sds)
    cols = ["Time","x","y","z","rx", "ry", "rz", "rw", "Released", "xt", "yt", "zt"]
    df = pd.DataFrame(data=trajectory, columns=cols)
    df, _ = quaternion_norm(df)
    #t = pd.Series(data=np.arange(0,5,0.01), name="Time")
    #output = pd.concat([t,df], axis=1)
    df.to_csv(OFILE+f"-{STARTINDEX}-{ID}.csv", index=False)
    validated_dataset.to_csv(OFILE.replace("models/", "models/validation/")+".csv", index=False)