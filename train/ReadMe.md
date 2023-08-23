# Gex2SGen Model - Training

## Description
* The pre-processed SMILES dataset used for training the SMILES-VAE model, the vocabulary files and 
  sample smiles induced gene-expression data for training and testing have been provided in the "data" folder

* The Gex2SGen.py takes as input:
        * file_l1000_train: Path to file containing preprocessed training data
        * file_l1000_test: Path to containing preprocessed test data
        * pathdir: Path to save the new models
        * model_name: User defined name of the model
        * SVAE_saved_path: Path to the SMILES-VAE saved model
        * PVAE_saved_path: Path to the PVAE saved model
        * char_to_int_file: Dictionary mapping the characters to integers
        * int_to_char_file: Dictionary mapping the integers to the characters

* Other Parameters: (Can be modified for user requirements. These hyperparameters are for reproducing our results)
        * SAVE_INTERVAL: Interval to save the models
        * BATCH_SIZE: size of batch for training data
        * BATCH_SIZE_TEST: size of batch for test data
        * LR: learning rate of model
        * EPOCHS: Number of epochs to run
        * KL_GROWTH_RATE: growth rate of KL divergence for loss calculations

* Output:
        * Models at intervals of <SAVE_INTERVAL> saved at <pathdir>/<model_name>
        * Loss obtained from each epoch printed in console.


* The files are passed to the program as command-line arguments (code can be modified to directly include the paths to the respective files).
* The checkpoint file with model state and optimizer state will be saved to a  user-defined folder once in every user-defined epochs.

## Prerequisites
* Anaconda or Miniconda with Python 3.6 or 3.8.
* CUDA-enabled GPU optional (Cannot be backpropagated without GPU - system can get stuck due to number of parameters involved).
* A conda environment with the following libraries:
        * pandas
        * numpy
        * PyTorch (1.0.0 or greater)
        * pickle
        * Generic libraries: sys, os

## Sample command - Training from scratch
* python Gex2SGen.py "/data/sample_L1000_train.txt" "/data/sample_L1000_test.txt" <pathdir> <model_name> <SMILES-VAE_saved_path> <PVAE_saved_path> "/data/char_to_int_dict_total.pkl" "/data/int_to_char_dict_total.pkl"


