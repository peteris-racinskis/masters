#!/usr/bin/env python3
import pandas as pd
import numpy as np
from os import listdir
from os.path import exists
from typing import Tuple
import tensorflow as tf
from helpers import data_and_label, generate_trajectory
from tensorflow.keras import layers, optimizers, losses, models
BASE_DIR="processed_data/norm/"
TIME="Time"
TEST="processed_data/train_datasets/test-start-time-doubled-7db3d40f19abc9f24f46.csv"
TRAIN="processed_data/train_datasets/train-start-time-doubled-7db3d40f19abc9f24f46.csv"
H_NEURONS=128
EPOCHS=300
OFILE=f"models/naiveBC-RNNx{H_NEURONS}x2-ep{EPOCHS}-norm-start-timesignal"
OVERWRITE=False
STARTINDEX=0
ID=""
huber = losses.Huber()

# Create a batch of initial states
# s0 s0 s0 .... s0
# feed into rnn to get s1
# update sequence to be
# s0 s0 s0 .... s1
# repeat to get
# s0 ... s1 s2 s3 ... sn

def generate_rnn_trajectory(model, initial_state: np.ndarray, length=50) -> np.ndarray:
    state_history = tf.constant(initial_state.reshape(1,1,-1), dtype=tf.float32)
    target = state_history[:,:,-3:]
    t_base = tf.reshape(tf.constant(0.01, dtype=tf.float32), (1,1,-1))
    for i in range(1,51):
        t  = t_base * i
        new_state = model(state_history)
        new_state = new_state[:,-1:,:]
        new_state = tf.concat([t, new_state, target], axis=2)
        state_history = tf.concat([state_history, new_state], axis=1)
    return tf.squeeze(state_history).numpy()


def generate_trajectories_with_target_rnn(model, means: np.ndarray, deviations: np.ndarray, num=50, length=50):
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
        trajectories.append(generate_rnn_trajectory(model, init, length))
        #trajectories.append(generate_trajectory(model, init, length))
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

def quaternion_norm(df: pd.DataFrame):
    quat_cols = ["r"+c for c in "xyzw"]
    quat_norm_series = df[quat_cols].pow(2).sum(axis=1).pow(1/2)
    quat_norm_series.name = "quaternion_norm"
    return pd.concat([df,quat_norm_series], axis=1), True

def create_sequential_dataset(df: pd.DataFrame):
    data_blocks, label_blocks = [], []
    startindices = df.loc[lambda d: d[TIME] == 0].index
    for i, ii in zip(startindices[:-1], startindices[1:]):
        data, label = data_and_label(df.iloc[i:ii], shuffle=False)
        data_blocks.append(data)
        label_blocks.append(label)
    data_rag_tensor = tf.ragged.constant(data_blocks, ragged_rank=1)
    label_rag_tensor = tf.ragged.constant(label_blocks, ragged_rank=1)
    return data_rag_tensor, label_rag_tensor

if __name__ == "__main__":
    np.random.seed(133)
    df = pd.read_csv(TRAIN)

    train_data, train_labels = create_sequential_dataset(df)

    target_means = tf.reduce_mean(train_data, axis=(0,1)).numpy()[-3:]
    target_sds = tf.math.reduce_std(train_data, axis=(0,1)).numpy()[-3:]
   
    if not exists(OFILE) or OVERWRITE:
        # Apparnetly, GRU is preferred for small datasets while
        # LSTM is preferred for large ones
        model = tf.keras.Sequential([
            layers.Input(shape=(None,) + train_data[0,0].shape),
            layers.GRU(H_NEURONS, return_sequences=True),
            layers.Dense(train_labels[0,0].numpy().size, activation=None),
        ])

        model.compile(
            optimizer=optimizers.Adam(10e-4),
            loss=huber
        ) 

        history = model.fit(
            train_data,
            train_labels,
            epochs=EPOCHS,
            validation_freq=1,
            verbose=1,
        )

        model.save(OFILE)
    else:
        model = models.load_model(OFILE)

    trajectory = generate_trajectories_with_target_rnn(model, target_means, target_sds)
    cols = ["Time","x","y","z","rx", "ry", "rz", "rw", "Released", "xt", "yt", "zt"]
    df = pd.DataFrame(data=trajectory, columns=cols)
    df, _ = quaternion_norm(df)
    df.to_csv(OFILE+f"-{STARTINDEX}-{ID}.csv", index=False)
    pass