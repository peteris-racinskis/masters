import tensorflow as tf
from tensorflow.keras import layers, Input
import numpy as np
import pandas as pd
LSTM_UNITS=128
# By default, rnn layer p

def make_rnn(data_element: np.ndarray, hidden=10):
    model = tf.keras.Sequential()
    model.add(Input(shape=(None,)+data_element.shape)) # the first col specifies timesteps (any)
    model.add(layers.LSTM(LSTM_UNITS)) 
    # by default, only the final accumulated output gets passed on.
    # this way, the next layer has a regular shaped tensor to expect!
    model.add(layers.Dense(hidden, activation="relu"))
    model.add(layers.Dense(1))
    model.summary()
    return model


if __name__ == "__main__":
    d = [0.0] * 11
    dd = np.asarray(d)
    rnn = make_rnn(dd)
    ddd = np.stack([dd]*5)
    result = rnn(ddd.reshape(-1,5,11))
    pass