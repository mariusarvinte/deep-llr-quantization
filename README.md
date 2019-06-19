# Deep Log-Likelihood Ratio (L-Value) Quantization
Common repository for the papers:
- *Deep Learning-Based L-Value Quantization for Gray-Coded Modulation*
- *Deep Log-Likelihood Ratio Quantization*

Marius Arvinte, Ahmed H. Tewfik and Sriram Vishwanath, University of Texas at Austin.

# Description
This repository contains source code for training and evaluating deep learning models for log-likelihood ratio (LLR, L-values) compression and finite precision quantization. For more details, please see our papers.

- Python requirements: 3.6+, Keras 2.2.4, Tensorflow 1.13.1, scikit-learn
- (Optional) Matlab requirements: R2014a+, Communication Toolbox

# Instructions
- (Optional) Use the function 'matlab/GenTrainingData.m' to generate .mat files containing training and test collections of LLR (L-values) in the format [num_snr, num_packets, codeword_length]

- Use 'deep_quantization_joint_decoding.py' to train and evaluate the performance of a joint-decoder architecture, as in the *Deep Log-Likelihood Ratio Quantization* paper.

- Use 'deep_quantization_marginal_decoding.py' to train and evaluate the performance of a branched-decoder architecture, as in the *Deep Learning-Based L-Value Quantization for Gray-Coded Modulation* paper.

- Both previous scripts will save a .mat file with the reconstructed LLR (L-values) in the 'data' folder.

- (Optional) Use the function 'matlab/TestReconstructedData.m' to decode using the reconstructed LLR (L-values) and get the Block Error Rate performance.
