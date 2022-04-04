import pandas as pd
import numpy as np
from os import listdir
from os.path import exists
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, models
from helpers import data_and_label, generate_trajectory, make_generator, make_discriminator
TRAIN="processed_data/train_datasets/train-003430811ff20c35ccd5.csv"
TEST="processed_data/train_datasets/test-003430811ff20c35ccd5.csv"
OFILE="models/naiveBC-small-movement-thresh"
OVERWRITE=False
STARTINDEX=156

@tf.function
def train_step(demos):
    pass



if __name__ == "__main__":
    np.random.seed(133)
    train_data, train_labels = data_and_label(pd.read_csv(TRAIN))
    test_data, test_labels = data_and_label(pd.read_csv(TEST))

    generator = make_generator(train_data[0], train_labels[0])
    generator.summary()

    discriminator = make_discriminator(train_labels[0])
    discriminator.summary()
