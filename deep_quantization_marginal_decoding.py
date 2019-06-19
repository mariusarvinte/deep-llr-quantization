# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:17:04 2019

@author: ma56473
"""

from keras.callbacks import TerminateOnNaN, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

from aux_networks import BranchedAutoencoder
from aux_networks import sample_balanced_wmse
from aux_networks import sample_wmse, sample_wmse_numpy

from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib

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
# Training schedule parameters
# Stage 1
num_rounds   = 10 # Corresponds to Nr in the paper
num_epochs_1 = 20 # Corresponds to Ne1 in the paper
# Stage 2
num_epochs_2 = 100 # Corresponds to Ne2 in the paper, same for all decoders
# Inference parameters
# NOTE: This will throw an error if your GPU memory is not sufficient
inf_batch_size = 65536

# K-means quantization parameters
num_bits = [6, 6, 6] # Fine-tunable per dimension

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

## Stage 1 - Periodically update the weights wk
# For each round
for round_idx in range(num_rounds):
    # Clear session
    K.clear_session()
    
    # Initial weights
    if round_idx == 0:
        # Initial weight tensor
        loss_np = np.asarray([1., 1., 1., 1., 1., 1., 1., 1.])
    
    # Normalize and update weights
    loss_weights = K.expand_dims(K.variable(loss_np / np.sum(loss_np)))
        
    # Instantiate blank autoencoder
    ae, ae_list, enc, dec, dec_list = BranchedAutoencoder(mod_size, latent_dim, num_layers, hidden_dim,
                                      common_layer, latent_layer, weight_l2_reg, local_seed,
                                      verbose=False, noise_sigma=noise_sigma)
    # Early stop
    earlyStop = EarlyStopping(monitor='val_loss', patience=100, min_delta=0.0001,
                              restore_best_weights=True)
    # Save best weights
    bestModel = ModelCheckpoint('models/ae_marginal_weights_best.h5',
                                verbose=0, save_best_only=True, save_weights_only=True, period=1)
    # Local optimizer
    optimizer = Adam(lr=0.001, amsgrad=True)
    # Compile with custom weighted loss function
    ae.compile(optimizer=optimizer, loss=sample_balanced_wmse(eps=global_eps, weights=loss_weights))
    
    # Load last round weights and optimizer state
    if round_idx > 0:
        ae._make_train_function()
        ae.optimizer.set_weights(weight_values)
        ae.load_weights('models/tmp_weights.h5')

    # Train
    history = ae.fit(x=llr_train, y=llr_train, batch_size=batch_size, epochs=num_epochs_1,
                               validation_split=0.2, verbose=2,
                               callbacks=[earlyStop, bestModel, TerminateOnNaN()])
    
    # Evaluate on training data
    rec_train = ae.predict(llr_train, batch_size=inf_batch_size)
    loss_np   = sample_wmse_numpy(llr_train, rec_train, eps=global_eps) # This is sufficient
    # Print errors
    print('Per-output error is:' + str(loss_np))
    
    # Save weights and optimizer state
    symbolic_weights = getattr(ae.optimizer, 'weights')
    weight_values = K.batch_get_value(symbolic_weights)
    ae.save_weights('models/tmp_weights.h5')
    
# Freeze encoder
enc.trainable = False
# Recompile with slower learning rate and WMSE
optimizer = Adam(lr=0.0005, amsgrad=True)
ae.compile(optimizer=optimizer, loss=sample_wmse(global_eps))

## Stage 2 - Continue training decoders
# New callbacks
# Early stop
earlyStop = EarlyStopping(monitor='val_loss', patience=100, min_delta=1e-5,
                          restore_best_weights=True)
# Save best weights
bestModel = ModelCheckpoint('models/ae_marginal_weights_best.h5',
                            verbose=0, save_best_only=True, save_weights_only=True, period=1)

# Train (fully parallel)
history = ae.fit(x=llr_train, y=llr_train, batch_size=batch_size, epochs=num_epochs_2,
                 validation_split=0.2, verbose=2,
                 callbacks=[earlyStop, bestModel, TerminateOnNaN()])
## Train k-Means quantizers on latent training data
# Reload best weights
ae.load_weights('models/ae_marginal_weights_best.h5')

# Compute latent representation of training data
latent_train = enc.predict(llr_train, batch_size=inf_batch_size)

# One quantizer per dimension
for dim_idx in range(latent_dim):
    # Fit
    kmeans = MiniBatchKMeans(n_clusters=2**num_bits[dim_idx], verbose=2,
                             batch_size=8192, n_init=1000, max_no_improvement=200)
    kmeans.fit(np.reshape(latent_train[:,dim_idx], (-1, 1)))
    # Save trained model to file
    joblib.dump(kmeans, 'models/marginal_kmeans_dimension%d_bits%d.sav' % (
            dim_idx, num_bits[dim_idx]))

### Inference
# After training, reload best weights
# Alternatively, can run only this portion if weights are available
ae.load_weights('models/ae_marginal_weights_best.h5')

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

# Quantize latent data
# If performance without quantization is desired skip next loop
latent_val_q = np.copy(latent_val)
# Apply k-Means quantization with saved models
for dim_idx in range(latent_dim):
    # Load pretrained model
    kmeans = joblib.load('models/marginal_kmeans_dimension%d_bits%d.sav' % (
            dim_idx, num_bits[dim_idx]))
    # Extract codebook
    codebook = kmeans.cluster_centers_
    # Predict codebook index
    codebook_idx = kmeans.predict(np.reshape(latent_val[:,dim_idx], (-1, 1)))
    # Assign values from codebook
    latent_val_q[:, dim_idx] = np.squeeze(codebook[codebook_idx])

# Recover soft bits with decoder
llr_val_rec = dec.predict(latent_val_q, batch_size=inf_batch_size)

# Reshape to original format and convert to LLR
llr_val_rec = np.reshape(2 * np.arctanh(llr_val_rec), val_format)

# Save to output .mat file
hdf5storage.savemat('data/reconstructed_llr_mod%d_bitseed%d' % (
        mod_size, val_bit_seed), {'rec_llr': llr_val_rec})