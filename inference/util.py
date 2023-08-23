# Utilities to prepare the SMILES dataset to be input to the model

import csv
import pandas as pd
import numpy as np
import pickle
import torch

from torch.utils import data
from io import open

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Given a SMILES input file read and generate an array of SMILES molecules
def generate_smiles_dataset_from_file(smilesfile):
    reg_smiles = []
    with open(smilesfile) as molecules:
        molreader = csv.reader(molecules, delimiter='\t',
                               quotechar='"')  # Assuming the input file is in TSV format ("\t" separated)
        for molecule in molreader:
            smiles = ''.join(molecule[0])  # The first column of the input file should contain the SMILES
            reg_smiles.append(smiles)
            # print(reg_smiles)
    return reg_smiles


# Take a SMILES string and return it in '*' tokenized form - To account for bisyllable atoms
def tokenize(smiles):
    bisyllable_atoms = ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar', 'Ca', 'Ti', 'Cr', 'Fe', 'Ni', 'Cu',
                        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
                        'Cd', 'Sb', 'Te', 'Xe', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Er',
                        'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Re', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'At', 'Fr', 'Ra',
                        'Ac', 'Th', 'Pa', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'Lr', 'Rf', 'Db', 'Sg', 'Mt',
                        'Ds', 'Rg', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Zn', 'Mn']

    smiles.strip()
    tokenized = ""
    myiter = iter(range(1, len(smiles)))
    for i in myiter:
        test = smiles[i - 1] + smiles[i]
        if test in bisyllable_atoms:
            tokenized = tokenized + test + "*"
            if i < len(smiles) - 1:
                next(myiter)
        elif test in (atom.lower() for atom in bisyllable_atoms):
            tokenized = tokenized + test + "*"
            if i < len(smiles) - 1:
                next(myiter)
        else:
            tokenized = tokenized + smiles[i - 1] + "*"
            if i == len(smiles) - 1:
                tokenized = tokenized + smiles[i] + "*"
    tokenized = tokenized[:-1]
    return tokenized


def prepare_smile_one_hot_encoding(smiles, embed, char_to_int_dict):
    tokenized_smiles = []
    for smile in smiles:
        tokenized_smiles.append(tokenize(smile))
    tokenized_smiles = np.array(tokenized_smiles)

    one_hot = np.zeros((len(smiles), embed), dtype=np.int64)
    for i, smile in enumerate(tokenized_smiles):
        one_hot[i, 0] = char_to_int_dict["!"]
        molchar = smile.split("*")
        for j, c in enumerate(molchar):
            one_hot[i, j + 1] = char_to_int_dict[c]
        one_hot[i, len(molchar) + 1:] = char_to_int_dict["E"]

    return one_hot[:, 0:-1], one_hot[:, 1:]


# Given a SMILES dataset tokenize, find the unique characters and prepare a two-way dictionary - Return all outputs
def prepare_dicts_and_dictfiles(smiles, char_to_int_file, int_to_char_file):
    tokenized_smiles = []
    for smile in smiles:
        tokenized_smiles.append(tokenize(smile))

    tokenized_smiles = np.array(tokenized_smiles)

    with open(char_to_int_file, 'rb') as handle:
        char_to_int = pickle.loads(handle.read())

    with open(int_to_char_file, 'rb') as handle:
        int_to_char = pickle.loads(handle.read())

    charset = char_to_int.keys()
    return tokenized_smiles, charset, char_to_int, int_to_char


# Given the SMILES vocabulary, prepare the character set of unique symbols and return them along with the dictionaries
def prepare_dicts_and_dictfiles_only(char_to_int_file, int_to_char_file):
    with open(char_to_int_file, 'rb') as handle:
        char_to_int = pickle.loads(handle.read())

    with open(int_to_char_file, 'rb') as handle:
        int_to_char = pickle.loads(handle.read())

    charset = char_to_int.keys()
    return charset, char_to_int, int_to_char


def prepare_smile_one_hot_encoding_prefind(smiles, embed, char_to_int_dict):
    tokenized_smiles = []
    for smile in smiles:
        tokenized_smiles.append(tokenize(smile))
    tokenized_smiles = np.array(tokenized_smiles)
    one_hot = np.zeros((len(smiles), embed), dtype=np.int64)

    for i, smile in enumerate(tokenized_smiles):
        one_hot[i, 0] = char_to_int_dict["!"]
        molchar = smile.split("*")
        for j, c in enumerate(molchar):
            one_hot[i, j + 1] = char_to_int_dict[c]
        one_hot[i, len(molchar) + 1:] = char_to_int_dict["E"]
    return one_hot


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 978))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    print('KLD', KLD.item(), 'BCE', BCE.item())
    return BCE + KLD, BCE, KLD


def reparameterization_trick(mu, sigma):
    std = torch.exp(0.5 * sigma)
    eps = torch.randn_like(std)
    latent_z = eps.mul(std).add_(mu)
    return latent_z


def kl_divergence_loss(mu, logvar, kl_growth=0.0015, step=None, eval_mode=False):
    """KL Divergence loss from VAE paper.

	Reference:
		Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014

	Args:
		mu (torch.Tensor): Encoder output of means of shape
			`[batch_size, input_size]`.
		logvar (torch.Tensor): Encoder output of logvariances of shape
			`[batch_size, input_size]`.
	Returns:
		The KL Divergence of the thus specified distribution and a unit
		Gaussian.
	"""
    mu = mu.to(device)
    logvar = logvar.to(device)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kld


def main():
    print("Hello World")


if __name__ == "__main__":
    main()
else:
    print('Just Imported')
