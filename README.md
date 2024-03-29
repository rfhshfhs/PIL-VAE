# PIL-VAE
# Can We Build a Generative Model without Back Propagation Training?

This project contains the implementation code for the paper "Can We Build a Generative Model without Back Propagation Training?" We provide the code in Matlab, TensorFlow, and PyTorch versions so that researchers and developers can choose the framework that best suits their needs.

## Project Structure

- `matlab/`: Contains the Matlab implementation.
- `tensorflow/`: Contains the TensorFlow implementation.
- `pytorch/`: Contains the PyTorch implementation.

## Requirements

- Matlab (Recommended version: R2020a)
- Python (Recommended version: 3.8)
- TensorFlow (Recommended version: 2.4)
- PyTorch (Recommended version: 1.8)

## Installation Guide

Ensure you have the correct version of the above software or libraries installed. For Python libraries, you can use `pip` to install them:

pip install tensorflow==2.4

pip install torch==1.8


# Running the Code

To run the code for a specific framework, enter the corresponding folder and execute the main script:

## For Matlab code:

cd matlab

matlab -r "VAE_pil0_PILgn_ppcamle_MNISTmulti.m"

## For TensorFlow code:

cd tensorflow

python pilvae-tensorflow.py

## For PyTorch code:

cd pytorch

python pilvae-pytorch.py
