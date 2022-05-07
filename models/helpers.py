from cProfile import label
from typing import Tuple
import pandas as pd
import numpy as np
from tensorflow.keras import layers, Input, Model
GEN_H=256
DISC_H=512
LSTM_H=128
RNN_FINAL_H=10


def make_generator(data_element: np.ndarray, label_element: np.ndarray, gen_h=GEN_H):
    inputs = Input(shape=data_element.shape)
    x = layers.Dense(gen_h)(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(gen_h)(x)
    x = layers.LeakyReLU()(x)
    outputs = layers.Dense(label_element.size)(x)
    return Model(inputs, outputs, name="generator")

def make_rnn_discriminator(data_element: np.ndarray):
    inputs = Input(shape=(None,)+data_element.shape)
    x = layers.LSTM(LSTM_H)(x)
    x = layers.Dense(RNN_FINAL_H)(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(1)(x)
    return Model(inputs, outputs, name="discriminator")

def make_discriminator(data_element: np.ndarray, label_element: np.ndarray, classes=2, disc_h=DISC_H):
    comb = np.concatenate([data_element, label_element])
    inputs = Input(shape=comb.shape)
    x = layers.Dense(disc_h)(inputs) # two parentheses - (constructor)(call)
    x = layers.ReLU()(x)
    x = layers.Dense(disc_h)(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(classes)(x)
    return Model(inputs, outputs, name="discriminator")

def data_and_label(df: pd.DataFrame, shuffle=True) -> Tuple[np.ndarray]:
    values = df.values.astype(np.float32)
    if shuffle:
        np.random.shuffle(values)
    data_cols = values[:,:-8]
    label_cols = values[:,-8:]
    return data_cols, label_cols

def generate_trajectory(model, initial_state, steps):
    states = []
    reshaped = initial_state.reshape(1,12)
    target = reshaped[:,-3:]
    state = reshaped
    for i in range(steps):
        t = np.asarray((i+1) * 0.01).reshape(-1,1)
        newstate = model(state).numpy()
        states.append(state)
        state = np.concatenate([t, newstate, target], axis=1)
    arr = np.asarray(states)
    return arr.reshape(-1,12)

def quaternion_norm(df: pd.DataFrame):
    quatnorm = "quaternion_norm"
    if quatnorm in df.columns:
        return df, False
    quat_cols = ["r"+c for c in "xyzw"]
    quat_norm_series = df[quat_cols].pow(2).sum(axis=1).pow(1/2)
    quat_norm_series.name = quatnorm
    return pd.concat([df,quat_norm_series], axis=1), True