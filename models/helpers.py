from cProfile import label
from typing import Tuple
import pandas as pd
import numpy as np
from tensorflow.keras import layers, Input, Model
GEN_H=256
DISC_H=256
LSTM_H=128
RNN_FINAL_H=10


def make_generator(data_element: np.ndarray, label_element: np.ndarray):
    inputs = Input(shape=data_element.shape)
    x = layers.Dense(GEN_H)(inputs)
    x = layers.ReLU()(x)
    x = layers.Dense(GEN_H)(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(label_element.size)(x)
    return Model(inputs, outputs, name="generator")

def make_rnn_discriminator(data_element: np.ndarray):
    inputs = Input(shape=(None,)+data_element.shape)
    x = layers.LSTM(LSTM_H)(x)
    x = layers.Dense(RNN_FINAL_H)(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(1)(x)
    return Model(inputs, outputs, name="discriminator")

def make_discriminator(data_element: np.ndarray, label_element: np.ndarray, classes=2):
    comb = np.concatenate([data_element, label_element])
    inputs = Input(shape=comb.shape)
    x = layers.Dense(DISC_H)(inputs) # two parentheses - (constructor)(call)
    x = layers.ReLU()(x)
    x = layers.Dense(DISC_H)(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(classes)(x)
    return Model(inputs, outputs, name="discriminator")

def data_and_label(df: pd.DataFrame) -> Tuple[np.ndarray]:
    values = df.values.astype(np.float32)
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
        t = np.asarray(i * 0.01).reshape(-1,1)
        state = model(state).numpy()
        state = np.concatenate([t, state, target], axis=1)
        states.append(state)
    arr = np.asarray(states)
    return arr.reshape(-1,12)