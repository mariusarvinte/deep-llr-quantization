# deep-llr-quantization
Source code for the "Deep Log-Likelihood Ratio Quantization" paper submitted to EUSIPCO 2019.

**Marius Arvinte, Ahmed H. Tewfik and Sriram Vishwanath**, University of Texas at Austin.

# Instructions

1. Run the Matlab file 'single_transmission.m' in 'Train' mode to generate LLR collection for autoencoder training.

2. Run the 'train_autoencoder' function in the Python 'deep_quantizer.py' file to train the autoencoder.

3. Run the Matlab files 'single_transmission.m' or 'multiple_transmission' in 'Test-out' mode to generate LLR collection for autoencoder testing.

4. Run the 'compress_llr' and 'quantize_reconstruct_llr' functions in the Python 'deep_quantizer.py' file to compress, quantize and reconstruct the log-likelihood ratios.

5. Run the Matlab files 'single_transmission.m' or 'multiple_transmission' in 'Test-in' mode to evaluate the performance.

Pretrained DNNs are available in the 'results' folder, allowing to skip steps 1-2.
