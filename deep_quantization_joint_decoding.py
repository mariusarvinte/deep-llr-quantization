# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:17:04 2019

@author: ma56473
"""

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

from aux_networks import JointAutoencoder
from aux_networks import sample_wmse, scalar_quantizer

import numpy as np
import hdf5storage
import os

# GPU allocation
K.clear_session()
tf.reset_default_graph()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";
# Tensorflow memory allocation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
K.tensorflow_backend.set_session(tf.Session(config=config))

### Parameters and initializations
# DNN/Scenario parameters
mod_size   = 8 # K = log2(M) in the paper, bits per QAM symbol
latent_dim = 3 # Number of sufficient statistics
num_layers = 4 # Total number of layers per encoder/decoder
hidden_dim = [4*mod_size, 4*mod_size,
              4*mod_size, 4*mod_size]
common_layer = 'relu' # Hidden activations
latent_layer = 'tanh' # Latent representation activation
weight_l2_reg = 0. # L2 weight regularization
# Noise standard deviation
noise_sigma = 1e-3
# Epsilon in the loss function
global_eps  = 1e-6
# Initial weight seeding - this allows for completely reproducible results
local_seed = np.random.randint(low=0, high=2**31-1)
np.random.seed(local_seed)

# Training parameters
batch_size = 65536
num_epochs = 1000
# Inference parameters
# NOTE: This will throw an error if your GPU memory is not sufficient
inf_batch_size = 65536

# Instantiate model
ae, enc, dec = JointAutoencoder(mod_size, latent_dim, num_layers, hidden_dim,
                                common_layer, latent_layer, weight_l2_reg, local_seed,
                                verbose=False, noise_sigma=noise_sigma)

### Training
# Target file for training/validation data
# Split is made at random 80/20 train/val from the same file
# Currently implemented with .mat, but can be anything
train_file = 'data/ref_llr_mod8_seed1234.mat'
contents   = hdf5storage.loadmat(train_file)
llr_train  = np.asarray(contents['ref_llr'])
# Reshape, convert to soft bits and shuffle
llr_train = np.reshape(np.tanh(llr_train / 2), (-1, mod_size))
np.random.shuffle(llr_train)

# Compile model with optimizer
optimizer = Adam(lr=0.001, amsgrad=True)
ae.compile(optimizer=optimizer, loss=sample_wmse(eps=1e-4))

# Callbacks
# Reduce LR on plateau
slowRate = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=60,
                             verbose=1, cooldown=50, min_lr=0.0001)
# Early stop
earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=600,
                          verbose=1, restore_best_weights=True)
# Save best weights
bestModel = ModelCheckpoint('models/ae_joint_weights_best.h5',
                            verbose=0, save_best_only=True, save_weights_only=True, period=1)

# Training
history = ae.fit(x=llr_train, y=llr_train, batch_size=batch_size, epochs=num_epochs,
                  validation_split=0.2, callbacks=[slowRate, bestModel, earlyStop], verbose=2)

### Inference
# After training, reload best weights
# Alternatively, can run only this portion if weights are available
ae.load_weights('models/ae_joint_weights_best.h5')

# Load validation data
# Can load your own .mat file of LLRs here
# Format of variable is (-1, mod_size)
# Each row corresponds to LLRs from the same QAM symbol
# This is easily done with our script by changing 'bit_seed' in Matlab
val_bit_seed = 4321
val_file = 'data/ref_llr_mod%d_seed%d.mat' % (mod_size, val_bit_seed)
contents = hdf5storage.loadmat(val_file)
llr_val  = np.asarray(contents['ref_llr'])
# Reshape and convert to soft bits
val_format = llr_val.shape
llr_val    = np.reshape(np.tanh(llr_val / 2), (-1, mod_size))

# Predict latent representation of validation data
latent_val = enc.predict(llr_val, batch_size=inf_batch_size)
# Apply scalar quantization
min_clip = -0.8 
max_clip = 0.8
num_bits = 6 # Number of bits per dimension (equal)
latent_val_q = scalar_quantizer(latent_val, num_bits, min_clip, max_clip)

# Recover soft bits with decoder
llr_val_rec = dec.predict(latent_val_q, batch_size=inf_batch_size)

# Reshape to original format and convert to LLR
llr_val_rec = np.reshape(2 * np.arctanh(llr_val_rec), val_format)

# Save to output .mat file
hdf5storage.savemat('data/reconstructed_llr_mod%d_bitseed%d' % (
        mod_size, val_bit_seed), {'rec_llr': llr_val_rec})