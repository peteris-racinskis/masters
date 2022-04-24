#!/usr/bin/env python3
import pandas as pd
import numpy as np
from os import listdir
from os.path import exists
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, models
from helpers import data_and_label, generate_trajectory, make_generator, make_discriminator
#TRAIN="processed_data/train_datasets/train-003430811ff20c35ccd5.csv"
#TEST="processed_data/train_datasets/test-003430811ff20c35ccd5.csv"
# Normalized data (I think?)
TRAIN="processed_data/train_datasets/train-start-time-doubled-5e9156387f59cb9efb35.csv"
OVERWRITE=True
CONTINUE=True
STARTINDEX=0
BATCHSIZE=64
EPOCHS=200
GH=128
DH=512
OFILE=f"models/BCO-{GH}x2-{DH}x2-start-timesignal-doubled-noreg-ep{EPOCHS}-b{BATCHSIZE}-norm"

# Convenience declaration
# apparently this implements the __call__ method, which means
# that this is the class instantiation wheras the calls below
# refer to this method.
cross_entropy = losses.BinaryCrossentropy(from_logits=True)

# Compare discriminator's predictions to a vector of 1's
# (try to fool it)
def generator_loss(predictions_on_fake):
    all_fooled = tf.ones_like(predictions_on_fake)
    return cross_entropy(all_fooled, predictions_on_fake)# + tf.math.maximum(diff * 10, 1)

# Penalize for not telling generated and actual training data apart
def discriminator_loss(predictions_on_real, predictions_on_fake):
    loss_on_real = cross_entropy(tf.ones_like(predictions_on_real), predictions_on_real)
    loss_on_fake = cross_entropy(tf.zeros_like(predictions_on_fake), predictions_on_fake)
    return loss_on_fake + loss_on_real

def generator_multi_iterate(gen, initial_state):
    prev, new = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True), tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    state = tf.reshape(initial_state[:12], [1,12])
    target = tf.reshape(initial_state[12-3:12], [1,3])
    timesteps = tf.convert_to_tensor(np.arange(0, BATCHSIZE*0.01, 0.01, dtype=np.float32).reshape(BATCHSIZE,1))
    timesteps = timesteps + initial_state[0]
    for i in tf.range(BATCHSIZE):
        t = tf.reshape(timesteps[i], [1,1])
        prev = prev.write(i, state)
        state = gen(state)
        new = new.write(i, state)
        state = tf.concat([t, state, target], axis=1)
    oldtensor = tf.squeeze(prev.stack())
    newtensor = tf.squeeze(new.stack())
    leadup = tf.concat([oldtensor, newtensor, tf.roll(newtensor,-1,0), tf.roll(newtensor,-2,0)], axis=1)
    return leadup[:-3]

# Real IRL appraoches use algos like actor critic - the output of the discriminator
# is used as the reward for a classical RL algorithm like actor-critic. Perhaps need to do
# that to increase performance?
@tf.function
def train_step(inits, data, labels):


    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generator_outputs = generator_multi_iterate(generator, inits[0])

        predict_on_real = discriminator(tf.concat([data,labels], axis=1))
        predict_on_fake = discriminator(generator_outputs)

        gen_loss = generator_loss(predict_on_fake)
        disc_loss = discriminator_loss(predict_on_real, predict_on_fake)
    
    gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grad = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_opt.apply_gradients(zip(gen_grad, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grad, discriminator.trainable_variables))
    return gen_loss, disc_loss

def train(data, labels, tgms, tgsd, epochs=20):
    # Create batches for initial states, next states
    batched_data = tf.data.Dataset.from_tensor_slices(data).batch(BATCHSIZE)
    labels = tf.data.Dataset.from_tensor_slices(labels).batch(BATCHSIZE)
    for epoch in range(epochs):
        initial_states = tf.data.Dataset.from_tensor_slices(data).shuffle(len(data)).batch(BATCHSIZE)
        batched_data = batched_data
        for i_batch, d_batch, l_batch in zip(initial_states, batched_data, labels):
            g, d = train_step(i_batch, d_batch, l_batch)
        print(f"Epoch: {epoch} g_loss: {g} d_loss {d}")
        if epoch % 7 == 0: # I dunno, minimize chances of weird oscillations?
            print("Generating trajectories with current policy...")
            generate_intermediate(generator, tgms, tgsd, epoch, d)
        
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


def generate_intermediate(generator, target_means, target_sds, ep, disc_loss):
    trajectory = generate_trajectories_with_target(generator, target_means, target_sds)
    cols = ["Time","x","y","z","rx", "ry", "rz", "rw", "Released", "xt", "yt", "zt"]
    df = pd.DataFrame(data=trajectory, columns=cols)
    dl = "{:.2f}".format(disc_loss)
    save_name = OFILE+f"-{STARTINDEX}-at-{ep}-dl-{dl}.csv"
    while exists(save_name):
        dl = dl+EPOCHS
        save_name = OFILE+f"-{STARTINDEX}-at-{ep}-dl-{dl}.csv"
    df.to_csv(save_name, index=False)



if __name__ == "__main__":

    train_data, train_labels = data_and_label(pd.read_csv(TRAIN))
    target_means = np.mean(train_data, axis=0)[12-3:12]
    target_sds = np.std(train_data, axis=0)[12-3:12]

    if not exists(OFILE+"-gen") or OVERWRITE or CONTINUE:
        np.random.seed(133)
        if CONTINUE:
            generator = models.load_model(OFILE+"-gen")
            discriminator = models.load_model(OFILE+"-disc")
            print("Restored models")
        else:
            generator = make_generator(train_data[0,:12], train_labels[0], gen_h=GH)
            discriminator = make_discriminator(train_data[0], train_labels[0], disc_h=DH)
        generator.summary()
        discriminator.summary()

        gen_opt = optimizers.Adam(10e-5)
        disc_opt = optimizers.Adam(10e-5)

        train(train_data, train_labels, target_means, target_sds, EPOCHS)
        generator.save(OFILE+"-gen")
        discriminator.save(OFILE+"-disc")    
    else:
        generator = models.load_model(OFILE+"-gen")
    
    trajectory = generate_trajectories_with_target(generator, target_means, target_sds)
    cols = ["Time","x","y","z","rx", "ry", "rz", "rw", "Released", "xt", "yt", "zt"]
    df = pd.DataFrame(data=trajectory, columns=cols)
    df.to_csv(OFILE+f"-{STARTINDEX}.csv", index=False)