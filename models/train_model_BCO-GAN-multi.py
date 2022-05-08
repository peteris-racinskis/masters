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
#TRAIN="processed_data/train_datasets/train-start-time-doubled-5e9156387f59cb9efb35.csv"
# Timesignal, no doubling, no prepend
#TRAIN="processed_data_old/train_datasets/train-start-time-5e9156387f59cb9efb35.csv"
#TEST="processed_data_old/train_datasets/test-start-time-5e9156387f59cb9efb35.csv"
## Timesignal, doubling, no prepend
#TRAIN="processed_data_old/train_datasets/train-start-time-doubled-5e9156387f59cb9efb35.csv"
#TEST="processed_data_old/train_datasets/test-start-time-doubled-5e9156387f59cb9efb35.csv"
# Timesignal, doubling, prepend
TRAIN="processed_data_old/train_datasets/train-start-time-doubled-5e9156387f59cb9efb35-prep.csv"
TEST="processed_data_old/train_datasets/test-start-time-doubled-5e9156387f59cb9efb35-prep.csv"
OVERWRITE=False
CONTINUE=False
STARTINDEX=0
BATCHSIZE=64
EPOCHS=200
GH=256
DH=256
REPEATS_IN_DATASET=3
PREPEND=True
p = "-prepend" if PREPEND else ""
d = "doubled-" if REPEATS_IN_DATASET == 3 else ""
TIME="Time"
OFILE=f"models/saved_models/BCO-leaky_g-{GH}x2-{DH}x2-start-timesignal-{d}ep{EPOCHS}-b{BATCHSIZE}{p}"
#OFILE=f"models/BCO-256x2-256x2-start-timesignal-doubled-noreg-ep200-b64-norm"


# Convenience declaration
# apparently this implements the __call__ method, which means
# that this is the class instantiation wheras the calls below
# refer to this method.
cross_entropy = losses.BinaryCrossentropy(from_logits=True)

# Compare discriminator's predictions to a vector of 1's
# (try to fool it)
def generator_loss(predictions_on_fake, generator_outputs):
    all_fooled = tf.ones_like(predictions_on_fake)
    #quat_reg = tf.reduce_sum(tf.square(tf.reduce_sum(tf.square(generator_outputs[:,3:7]), axis=-1) - 1))

    #position_term = tf.reduce_sum(tf.math.reduce_euclidean_norm(y_true[:,:3] - y_pred[:,:3], keepdims=True))
    #orientation_term = tf.reduce_sum(tf.math.reduce_euclidean_norm(y_true[:,3:7] - y_pred[:,3:7], keepdims=True))
    #release_term = entropy(y_true[:,-1], y_pred[:,-1])
    return cross_entropy(all_fooled, predictions_on_fake) #+ 0.1 * quat_reg# + tf.math.maximum(diff * 10, 1)

# Penalize for not telling generated and actual training data apart
def discriminator_loss(predictions_on_real, predictions_on_fake):
    loss_on_real = cross_entropy(tf.ones_like(predictions_on_real), predictions_on_real)
    loss_on_fake = cross_entropy(tf.zeros_like(predictions_on_fake), predictions_on_fake)
    return loss_on_fake + loss_on_real

def generator_multi_iterate(gen, initial_state):
    prev, new = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True), tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    state = tf.reshape(initial_state[:12], [1,12])
    mod_state = tf.identity(state)
    target = tf.reshape(initial_state[12-3:12], [1,3])

    spoolup_time = tf.convert_to_tensor(np.arange(-3, 0, 1, dtype=np.float32).reshape(3,1))
    for i in tf.range(3):
        t = tf.reshape(spoolup_time[i], [1,1])
        mod_state = tf.concat([t, mod_state[:,1:]], axis=1)
        prev = prev.write(i, mod_state)
        new = new.write(i, mod_state[:,1:9])

    timesteps = tf.convert_to_tensor(np.arange(0, BATCHSIZE*0.01, 0.01, dtype=np.float32).reshape(BATCHSIZE,1))
    timesteps = timesteps + initial_state[0]
    for i in tf.range(3, BATCHSIZE):
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

        gen_loss = generator_loss(predict_on_fake, generator_outputs)
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
        if epoch % 17 == 0: # I dunno, minimize chances of weird oscillations?
            print("Generating trajectories with current policy...")
            generate_intermediate(generator, tgms, tgsd, epoch, d)

def validation_on_test(df: pd.DataFrame, model, prep=PREPEND):
    stop_offset = -1 * REPEATS_IN_DATASET if prep else 0
    start_indices = list(df.loc[lambda d: d[TIME] == 0].index)
    start_indices += [df.index[-1] - stop_offset + 2]
    demos_with_trajectories = []
    column_count = 12
    generated_slice = slice(1,-3) 
    columns = list(df.columns[:column_count])
    columns += [c for c in "xyz"]
    columns += ["r"+c for c in "xyzw"]
    columns += ["Released-model"]
    for i, ii in zip(start_indices[:-1], start_indices[1:]):
        demo, _ = data_and_label(df.iloc[i:ii+stop_offset], shuffle=False)
        demo = demo[:,:12] # remove the nextstates
        initial_state = demo[0:1]
        trajectory = generate_trajectory(model, initial_state, len(demo))
        demos_with_trajectories.append(np.concatenate([demo, trajectory[:,generated_slice]], axis=1))
    combined_data = np.concatenate(demos_with_trajectories, axis=0)
    combined_df = pd.DataFrame(data=combined_data, columns=columns)
    return combined_df

def generate_trajectories_with_target(model, means: np.ndarray, deviations: np.ndarray, num=50, length=50):
    trajectories = []
    means = np.repeat(means.reshape(1,-1), num, axis=0)
    deviations = np.repeat(deviations.reshape(1,-1), num, axis=0)
    target_coords = np.random.normal(means, deviations, means.shape)
    #tc = np.asarray([-2.5049639 , -0.03949555, -0.30162135])
    #target_coords = np.repeat(tc.reshape(1,-1), num, axis=0)
    #init_orientation = np.asarray([-0.3134110987186432,0.6126522928476333,0.3159722849726677,0.6526271522045135]) # a random start orientation from the train dataset
    init_orientation = np.asarray([-0.188382297754288, 0.70863139629364, 0.236926048994064, 0.57675164937973]) # same thing the robot code uses
    #init_orientation = np.asarray([-0.18197935000062, 0.750300034880638, 0.247823745757341, 0.578429348766804])
    init_orientations = np.repeat(init_orientation.reshape(1,-1), num, axis=0)
    initial_states = np.concatenate([np.zeros((num,4)), init_orientations, np.zeros((num,1)), target_coords], axis=1) # stick a known good orientation in the set
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
    validated_on_test = validation_on_test(pd.read_csv(TEST), generator)
    validated_on_train = validation_on_test(pd.read_csv(TRAIN), generator)
    validated_on_test.to_csv(OFILE.replace("models/saved_models/", "models/validation/")+f"-at-{ep}-testval.csv", index=False)
    validated_on_train.to_csv(OFILE.replace("models/saved_models/", "models/validation/")+f"-at-{ep}-trainval.csv", index=False)


def quaternion_norm(df: pd.DataFrame):
    quat_cols = ["r"+c for c in "xyzw"]
    quat_norm_series = df[quat_cols].pow(2).sum(axis=1).pow(1/2)
    quat_norm_series.name = "quaternion_norm"
    return pd.concat([df,quat_norm_series], axis=1)


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
    df, _ = quaternion_norm(df)
    df.to_csv(OFILE+f"-{STARTINDEX}.csv", index=False)
    validated_on_test = validation_on_test(pd.read_csv(TEST), generator)
    validated_on_train = validation_on_test(pd.read_csv(TRAIN), generator)
    validated_on_test.to_csv(OFILE.replace("models/saved_models/", "models/validation/")+"-testval.csv", index=False)
    validated_on_train.to_csv(OFILE.replace("models/saved_models/", "models/validation/")+"-trainval.csv", index=False)