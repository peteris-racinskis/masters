from cProfile import label
from typing import Tuple
import pandas as pd
import numpy as np
from tensorflow.keras import layers, Input, Model
GEN_H=128
DISC_H=128


def make_generator(data_element: np.ndarray, label_element: np.ndarray):
    inputs = Input(shape=data_element.shape)
    x = layers.Dense(GEN_H)(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(GEN_H)(x)
    x = layers.LeakyReLU()(x)
    outputs = layers.Dense(label_element.size)(x)
    return Model(inputs, outputs, name="generator")

def make_discriminator(label_element: np.ndarray, classes=2):
    doubled = np.concatenate([label_element, label_element])
    inputs = Input(shape=doubled.shape)
    x = layers.Dense(GEN_H)(inputs) # two parentheses - (constructor)(call)
    x = layers.LeakyReLU()(x)
    outputs = layers.Dense(classes)(x)
    return Model(inputs, outputs, name="discriminator")

def data_and_label(df: pd.DataFrame) -> Tuple[np.ndarray]:
    values = df.values
    np.random.shuffle(values)
    data_cols = values[:,:-8]
    label_cols = values[:,-8:]
    return data_cols, label_cols

def generate_trajectory(model, initial_state, steps, states=[]):
    reshaped = initial_state.reshape(1,11)
    target = reshaped[:,-3:]
    state = reshaped
    for _ in range(steps):
        state = model(state).numpy()
        state = np.concatenate([state, target], axis=1)
        states.append(state)
    arr = np.asarray(states)
    return arr.reshape(-1,11)