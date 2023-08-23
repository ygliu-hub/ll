# Gex2SGen Model - Inference

## Description
* Sample gene expression file and the dictionaries have been provided in "data" folder

* The Gex2SGen_inference.py Method takes as input:
        * file to save the smiles
        * file containing inference data
        * name of model provided during training
        * Path to the PVAE saved model
        * Path to saved Gex2SGen model
        * char_to_int_file: Dictionary mapping the characters to integers
        * int_to_char_file: Dictionary mapping the integers to the characters
        * number of molecules to be generated

* Output:
        * Generated SMILES will be written in file <name_of_model>_<outfile_name>


* The files are passed to the program as command-line arguments (code can be modified to directly include the paths to the respective files).

* The output file will contain the generated SMILES

## Prerequisites
* Anaconda or Miniconda with Python 3.6 or 3.8.
* A conda environment with the following libraries:
        * pandas
        * numpy
        * PyTorch (1.0.0 or greater)
        * pickle
        * Generic libraries: sys, re, csv, os

## Sample command - Training from scratch
* python Gex2SGen_inference.py <file_to_save_SMILES> "/data/sample_inference.txt" <model_name> <PVAE_saved_path> <Gex2SGen_saved_path> "/data/char_to_int_dict_total.pkl" "/data/int_to_char_dict_total.pkl" <number_of_molecules>

