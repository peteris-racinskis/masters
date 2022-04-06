#!/usr/bin/env python3
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
BATCHSIZE=256

# Convenience declaration
# apparently this implements the __call__ method, which means
# that this is the class instantiation wheras the calls below
# refer to this method.
cross_entropy = losses.BinaryCrossentropy(from_logits=True)

# Compare discriminator's predictions to a vector of 1's
# (try to fool it)
def generator_loss(predictions_on_fake):
    all_fooled = tf.ones_like(predictions_on_fake)
    return cross_entropy(all_fooled, predictions_on_fake)

# Penalize for not telling generated and actual training data apart
def discriminator_loss(predictions_on_real, predictions_on_fake):
    loss_on_real = cross_entropy(tf.ones_like(predictions_on_real), predictions_on_real)
    loss_on_fake = cross_entropy(tf.zeros_like(predictions_on_fake), predictions_on_fake)
    return loss_on_fake + loss_on_real



@tf.function
def train_step(data, labels):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        state_transitions = generator(data)

        predict_on_real = discriminator(tf.concat([data,labels], axis=1))
        predict_on_fake = discriminator(tf.concat([data,state_transitions], axis=1))

        gen_loss = generator_loss(predict_on_fake)
        disc_loss = discriminator_loss(predict_on_real, predict_on_fake)
    
    gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grad = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_opt.apply_gradients(zip(gen_grad, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grad, discriminator.trainable_variables))
    return gen_loss, disc_loss

def train(data, labels, epochs=20):
    # Create batches for initial states, next states
    data = tf.data.Dataset.from_tensor_slices(data).batch(BATCHSIZE)
    labels = tf.data.Dataset.from_tensor_slices(labels).batch(BATCHSIZE)
    for epoch in range(epochs):
        for d_batch, l_batch in zip(data, labels):
            g, d = train_step(d_batch, l_batch)
        print(f"Epoch: {epoch} g_loss: {g} d_loss {d}")
        



if __name__ == "__main__":
    np.random.seed(133)
    train_data, train_labels = data_and_label(pd.read_csv(TRAIN))
    test_data, test_labels = data_and_label(pd.read_csv(TEST))

    generator = make_generator(train_data[0], train_labels[0])
    generator.summary()

    discriminator = make_discriminator(train_data[0], train_labels[0])
    discriminator.summary()

    gen_opt = optimizers.Adam(10e-4)
    disc_opt = optimizers.Adam(10e-4)

    train(train_data, train_labels, 200)


    pass