# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:17:04 2019

@author: ma56473
"""

from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adadelta
from keras.initializers import glorot_uniform
import numpy as np
import scipy.io as sio
from tqdm import tqdm

# Seed and train the autoencoder
def train_autoencoder(modSize, snrMin, snrMax, local_seed, batch_size=2**16, num_epochs=6000):
    # Load .mat files containing Matlab-generated training collection
    contents = sio.loadmat('MAT_Rayleigh_Train_mod' + str(modSize) + '_snr'
                           + str(snrMin) + 'to' + str(snrMax) + '.mat')
    llrCollect = contents['llrCollect']
    # Convert to soft bits
    llrCollect = np.tanh(np.reshape(llrCollect, (-1, modSize)) / 2)
    
    # NN output sizes
    input_dim = modSize
    hidden_dim = 4*modSize
    latent_dim = 3
    
    # Seeding the weights of each layer - always use 8 layers
    np.random.seed(local_seed)
    seed_array = np.random.randint(low=0, high=2**31, size=8)
    # Initializers
    weight_init = []
    for layer_idx in range(8):
        weight_init.append(glorot_uniform(seed=seed_array[layer_idx]))
        
    # Input layer
    input_bits = Input(shape=(modSize,))
    # Encoder
    encoded = Dense(hidden_dim, activation='relu', kernel_initializer=weight_init[0])(input_bits)
    encoded = Dense(hidden_dim, activation='relu', kernel_initializer=weight_init[1])(encoded)
    encoded = Dense(hidden_dim, activation='relu', kernel_initializer=weight_init[2])(encoded)
    encoded = Dense(latent_dim, activation='tanh', kernel_initializer=weight_init[3])(encoded)
    # Add (quantization) noise - edit power here if desired
    encoded_noisy = GaussianNoise(stddev=1.0/np.sqrt(1e6))(encoded)
    # Decoder
    decoded = Dense(hidden_dim, activation='relu', kernel_initializer=weight_init[4])(encoded_noisy)
    decoded = Dense(hidden_dim, activation='relu', kernel_initializer=weight_init[5])(decoded)
    decoded = Dense(hidden_dim, activation='relu', kernel_initializer=weight_init[6])(decoded)
    output_bits = Dense(input_dim, activation='tanh', kernel_initializer=weight_init[7])(decoded)
    
    # Models
    autoencoder = Model(input_bits, output_bits)
    # Save encoder/decoder models
    encoder = Model(input_bits, encoded)
    decoder_input = Input(shape=(latent_dim,))
    decoder = decoder_input
    # stack decoder layers
    for layer in autoencoder.layers[-4:]:
        decoder = layer(decoder)
    # create the decoder model
    decoder = Model(inputs=decoder_input, outputs=decoder)
    
    # Training algorithm
    adadelta = Adadelta(lr=2.0)
    
    # Print model summary
    autoencoder.summary()
    
    # Final model
    autoencoder.compile(optimizer=adadelta, loss='sample_weighted_mse')
    
    # Callbacks
    filestub = '_Rayleigh_mod' + str(modSize) + '_seed' + str(local_seed) + '_snr' + str(snrMin) + 'to' + str(snrMax)

    # Reduce LR on plateau
    slowRate = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=60,
                                 verbose=1, cooldown=50, min_lr=0.01)
    
    # Early stop
    earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=600,
                              verbose=1, restore_best_weights=True)
    
    # Training
    history = autoencoder.fit(x=llrCollect, y=llrCollect, batch_size=batch_size, epochs=num_epochs,
                      validation_split=0.2, callbacks=[slowRate, earlyStop], verbose=2)
    # Save global history
    train_loss = history.history['loss']
    val_loss   = history.history['val_loss']
    
    # Save last epoch model
    autoencoder.save('ae' + filestub + '.h5')
    encoder.save('enc' + filestub + '.h5')
    decoder.save('dec' + filestub + '.h5')
    
    return train_loss, val_loss

# Import collection of LLRs and output the latent representation and unquantized reconstruction
def compress_llr(modSize, snrMin, snrMax, local_seed):
    # Load pre-trained models
    autoencoder = load_model('ae_Rayleigh_mod' + str(modSize) + '_seed' + str(local_seed) + '_snr' + str(snrMin) + 'to' + str(snrMax)  + '.h5')
    encoder     = load_model('enc_Rayleigh_mod' + str(modSize) + '_seed' + str(local_seed) + '_snr' + str(snrMin) + 'to' + str(snrMax)  + '.h5')
    
    # Load .mat files
    contents = sio.loadmat('MAT_Rayleigh_Test_mod' + str(modSize) + '_snr' + str(snrMin) + 'to' + str(snrMax) +'.mat')
    # Convert to soft bits
    llrCollect = contents['llrCollect']
    llrShape   = llrCollect.shape
    bitBatches = np.tanh(np.reshape(llrCollect, (llrShape[0], -1, modSize)) / 2)

    # Reconstruct (no quantization) and compress
    rec_llrCollect  = np.zeros((bitBatches.shape))
    rec_llrCompress = np.zeros([bitBatches.shape[0], bitBatches.shape[1], 3])
    
    # For each SNR point compress and reconstruct (using entire autoencoder directly)
    for snr_idx in tqdm(range(llrShape[0])):
        rec_llrCollect[snr_idx,:,:]  = autoencoder.predict(bitBatches[snr_idx,:,:])
        rec_llrCompress[snr_idx,:,:] = encoder.predict(bitBatches[snr_idx,:,:])
        
    # Reshape back to suitable format
    rec_llrCollect = np.reshape(rec_llrCollect, (llrShape))
    # Revert reconstruction to log-domain
    rec_llrCollect = 2*np.arctanh(rec_llrCollect)
    
    # Save reconstruction - used to plot baseline AE performance (no quantization)
    sio.savemat('PY_Rayleigh_FP_mod' + str(modSize) + '_snr' + str(snrMin) + 'to' + str(snrMax) + '.mat',
                {'rec_llrCollect':rec_llrCollect})
    # Save compressed latent representation - this will be quantized (can be also done externally)
    sio.savemat('PY_Rayleigh_latent_mod' + str(modSize) + '_snr' + str(snrMin) + 'to' + str(snrMax) + '.mat',
                {'rec_llrCompress':rec_llrCompress})
    
# Simple scalar quantizer for the latent space
def scalar_quantizer(input_latent, dim_bits, min_clip, max_clip):
    # Generate quantization array
    qValues = np.expand_dims(np.linspace(start=min_clip, stop=max_clip, num=2**dim_bits), axis=0)
    
    # Output
    input_shape = input_latent.shape
    
    # Serialize input
    input_ser = np.expand_dims(input_latent.flatten(), axis=1)
    
    # L2 distance matrix
    distMatrix = np.abs(input_ser - qValues) ** 2
    # Nearest index
    nearestIdx = np.argmin(distMatrix, axis=1)
    # Convert to quantized value
    out_q = qValues[0, nearestIdx]
    # Reshape
    out_q = np.reshape(out_q, input_shape)
    
    return out_q

# Load latent representation, quantize and reconstruct LLRs with the decoder
def quantize_reconstruct_llr(modSize, snrMin, snrMax, local_seed, dim_bits, min_clip, max_clip):
    # Load decoder model
    decoder = load_model('dec_Rayleigh_mod' + str(modSize) + '_seed' + str(local_seed) + '_snr' + str(snrMin) + 'to' + str(snrMax)  + '.h5')
    
    # Load .mat files containing unquantized latent representation
    contents = sio.loadmat('PY_Rayleigh_latent_mod' + str(modSize) + '_snr' + str(snrMin) + 'to' + str(snrMax) + '.mat')
    rec_llrCompress = contents['rec_llrCompress']
    llrShape = rec_llrCompress.shape
    bitShape = (llrShape[0], int(llrShape[1]*llrShape[2]/modSize), modSize)
    
    # Apply quantization
    rec_llrCompressQ = scalar_quantizer(rec_llrCompress, dim_bits, min_clip, max_clip)
    
    # Reconstruct from quantized latent representation
    rec_llrCollectQ = np.zeros(bitShape)
    for snr_idx in tqdm(range(llrShape[0])):
        rec_llrCollectQ[snr_idx,:,:] = decoder.predict(rec_llrCompressQ[snr_idx,:,:])
        
    # Reshape to packet format
    rec_llrCollectQ = np.reshape(rec_llrCollectQ, llrShape)
    # Convert to log-domain
    rec_llrCollectQ = 2*np.arctanh(rec_llrCollectQ)
    
    # Save to .mat file
    sio.savemat('PY_Rayleigh_SQ_' + str(dim_bits) + 'bits_mod' + str(modSize) + '_snr' + str(snrMin) + 'to' + str(snrMax) +
            '.mat', {'rec_llrCollectQ':rec_llrCollectQ})
    

# Local seed - allows reproducible training of the autoencoder
# Use 832580347 to exactly reproduce the results in the paper
local_seed = np.random.randint(low=0, high=2**31)

# log2(M) for M-QAM
modSize = 8

# Training parameters
snrMin = 16
snrMax = 20

# latent space quantizer parameters
dim_bits = 4 # Bits per latent space dimension. Total storage size is 3x this.
min_clip = -0.8
max_clip = 0.8 # delta in the paper

# Train autoencoder
train_autoencoder(modSize, snrMin, snrMax, local_seed)
# Generate latent representation
compress_llr(modSize, snrMin, snrMax, local_seed)
# Quantize and reconstruct LLR
quantize_reconstruct_llr(modSize, snrMin, snrMax, local_seed, dim_bits, min_clip, max_clip)