# This Script is to generate SMILES from an Inference
import sys
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import torch.nn as nn
from torch import optim
from model import VAE_Encoder, VAE_Decoder, VAE
import utils

out_file_name = sys.argv[1]
file_l1000_inference = sys.argv[2]
model_name = sys.argv[3]
PVAE_saved_path = sys.argv[4]
Gex2SMILES_saved_path = sys.argv[5]
char_to_int_file = sys.argv[6]
int_to_char_file = sys.argv[7]
NUMBER_MOLECULES = sys.argv[8]

file_smiles_write = open(model_name + '_' + out_file_name, 'w')

BATCH_SIZE = 1
INPUT_SIZE = X_DIM = 978
OUTPUT_SIZE = INPUT_SIZE
LATENT_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STACK_WIDTH = 256
STACK_DEPTH = embed = 105
LR = 1e-3
GRU_DIM = 1024

print('Model Name', model_name)
print('PVAE_saved_path', PVAE_saved_path)
print('BATCH_SIZE', BATCH_SIZE, 'STACK_WIDTH', STACK_WIDTH, 'STACK_DEPTH', STACK_DEPTH, 'LR', LR, 'GRU_DIM', GRU_DIM)

charset, char_to_int, int_to_char = utils.prepare_dicts_and_dictfiles_only(char_to_int_file, int_to_char_file)
vocab_size = len(charset)


# Data Definition
class Dataset(Dataset):
    def __init__(self, file_l1000):
        self.L1000_data_file = file_l1000
        data = np.genfromtxt(self.L1000_data_file, delimiter=",", dtype=str, invalid_raise=False, autostrip=True,
                             comments=None, skip_header=0)
        self.smiles = data[:, 0]
        self.gex = torch.from_numpy(data[:, 1:].astype(float))  # The rest columns except the first 1 cols
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.gex)

    def __getitem__(self, index):
        Y_gex = self.gex[index]
        Y_smiles = self.smiles[index]
        return Y_smiles, Y_gex


inference_dataset = Dataset(file_l1000_inference)
data_loader_inference = torch.utils.data.DataLoader(inference_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                    drop_last=True)
print('Data Loading Done')

# Model Definition
decoder = VAE_Decoder(vocab_size, GRU_DIM, LATENT_SIZE, vocab_size, BATCH_SIZE, STACK_DEPTH, STACK_WIDTH, dropout=0.2,
                      n_layers=2, bidir=False).to(DEVICE)
pvae_encoder = VAE(INPUT_SIZE, LATENT_SIZE, OUTPUT_SIZE).to(DEVICE)

pvae_optimizer = optim.Adam(list(pvae_encoder.parameters()), lr=LR, amsgrad=True)
criterion = nn.CrossEntropyLoss()

# LOAD MODELS
pvae_state = torch.load(PVAE_saved_path, map_location='cpu')
print('PVAE Epoch:', pvae_state['epoch'])
pvae_encoder.load_state_dict(pvae_state['state_dict'])
pvae_optimizer.load_state_dict(pvae_state['optimizer'])
print(pvae_encoder)

Gex2SMILES_saved_path_m = Gex2SMILES_saved_path + '_' + model_name + '.pth'
print(Gex2SMILES_saved_path_m)
state_old = torch.load(Gex2SMILES_saved_path_m, map_location='cpu')
decoder.load_state_dict(state_old['state_dict'])

pvae_encoder.eval()
decoder.eval()
input_length = 99

print('Generating', NUMBER_MOLECULES, 'Molecules')
for batch_idx, (condition_inference, genex) in enumerate(data_loader_inference):
    print(batch_idx, condition_inference, 'Generating SMILES')
    decoder_stack = decoder.decoder_stackgru.initStack()
    genex = genex.view(BATCH_SIZE, X_DIM).to(DEVICE)

    mean, logvar = pvae_encoder.encode(genex.float())
    latent_z = utils.reparameterization_trick(mean, logvar)

    decoder_hidden = decoder.latent_vector_to_hidden(latent_z)
    try:
        for nmol in range(NUMBER_MOLECULES):
            prefix = torch.tensor(char_to_int["!"], device=DEVICE)
            smiles_str = ""
            for k in range(input_length):
                output, decoder_hidden, decoder_stack = decoder(prefix, decoder_hidden, decoder_stack)
                probs = torch.softmax(output, dim=2)
                top_i = torch.multinomial(probs.view(-1), 1)[0].to(DEVICE)
                top_i = top_i.item()
                predicted_char = int_to_char[top_i]

                if predicted_char == 'E':
                    break
                else:
                    smiles_str += predicted_char
                    prefix = torch.tensor(char_to_int[predicted_char], device=DEVICE)
                    prefix = prefix.unsqueeze(0).to(torch.int64)
            file_smiles_write.write(
                Gex2SMILES_saved_path_m + '\t' + str(batch_idx) + '\t' + condition_inference[0] + '\t' + str(nmol) + '\t' + smiles_str + '\n')
            if nmol % 100 == 0:
                print('No of Molecules Generated for', condition_inference, nmol, '/', NUMBER_MOLECULES)
    except Exception as e:
        print('Exception Caught:', e)

print('Completed Train & Test')
file_smiles_write.close()
