# This code trains the Gex2SGen model
import numpy as np
import torch
from torch import optim
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn import functional as F
import os
import sys
from model import Stack_GRU, VAE_Encoder, VAE_Decoder, VAE
import utils

file_l1000_train = sys.argv[1]
file_l1000_test = sys.argv[2]
pathdir = sys.argv[3]
model_name = sys.argv[4]
SVAE_saved_path = sys.argv[5]
PVAE_saved_path = sys.argv[6]
char_to_int_file = sys.argv[7]
int_to_char_file = sys.argv[8]

SAVE_INTERVAL = 10
BATCH_SIZE = 64
BATCH_SIZE_TEST = 64
LR = 1e-3
EPOCHS = 500
KL_GROWTH_RATE = 0.05

INPUT_SIZE = X_DIM = 978
OUTPUT_SIZE = INPUT_SIZE
LATENT_SIZE = 256
STACK_WIDTH = 256
STACK_DEPTH = embed = 105
GRU_DIM = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('CODENAME', model_name)
print('pathdir', pathdir)
print('SVAE_saved_path', SVAE_saved_path)
print('PVAE_saved_path', PVAE_saved_path)
print('BATCH_SIZE', BATCH_SIZE, 'STACK_WIDTH', STACK_WIDTH, 'STACK_DEPTH', STACK_DEPTH, 'LR', LR, 'GRU_DIM', GRU_DIM, 'KL_GROWTH_RATE', KL_GROWTH_RATE)

os.mkdir(pathdir + model_name)
PATH_TO_SAVE = pathdir + model_name + '/'
np.random.seed(42)

# Load char2Int, Int2Char, Charset
charset, char_to_int, int_to_char = utils.prepare_dicts_and_dictfiles_only(char_to_int_file, int_to_char_file)
vocab_size = len(charset)


# Data Definition
class Dataset(Dataset):
    def __init__(self, file_l1000):
        self.L1000_data_file = file_l1000
        data = np.genfromtxt(self.L1000_data_file, delimiter=",", dtype=str, invalid_raise=False, autostrip=True,
                             comments=None, skip_header=1)
        self.smiles = data[:, 0]
        self.smiles_one_hot = utils.prepare_smile_one_hot_encoding_prefind(self.smiles, embed, char_to_int)
        self.gex = torch.from_numpy(data[:, 1:].astype(float))  # The rest columns except the first 1 cols
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.gex)

    def __getitem__(self, index):
        Y_gex = self.gex[index]
        Y_smiles = self.smiles_one_hot[index]
        return Y_smiles, Y_gex


train_dataset = Dataset(file_l1000_train, )
test_dataset = Dataset(file_l1000_test)
data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=True, drop_last=True)
print('Data Loading Done')

# Model Definition
encoder = VAE_Encoder(vocab_size, GRU_DIM, LATENT_SIZE, vocab_size, BATCH_SIZE, STACK_DEPTH, STACK_WIDTH, dropout=0.2, n_layers=2, bidir=True).to(DEVICE)
decoder = VAE_Decoder(vocab_size, GRU_DIM, LATENT_SIZE, vocab_size, BATCH_SIZE, STACK_DEPTH, STACK_WIDTH, dropout=0.2, n_layers=2, bidir=False).to(DEVICE)
pvae_encoder = VAE(INPUT_SIZE, LATENT_SIZE, OUTPUT_SIZE).to(DEVICE)

# Optimizers
svae_optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR, amsgrad=True)
pvae_optimizer = optim.Adam(list(pvae_encoder.parameters()), lr=LR, amsgrad=True)
criterion = nn.CrossEntropyLoss()  # The reconstruction loss of the VAE joint loss function is Cross-entropy loss

# LOAD MODELS
pvae_state = torch.load(PVAE_saved_path, map_location='cpu')
pvae_encoder.load_state_dict(pvae_state['state_dict'])
pvae_optimizer.load_state_dict(pvae_state['optimizer'])
state_old = torch.load(SVAE_saved_path, map_location='cpu')
decoder.load_state_dict(state_old['decoder_state_dict'])
svae_optimizer.load_state_dict(state_old['optimizer'])

end_epoch = state_old['epoch']
end_train_loss = state_old['train_loss']
end_train_acc = state_old['train_acc']
end_val_loss = state_old['val_loss']
end_val_acc = state_old['val_acc']
print("Epoch_no: ", end_epoch, "Training loss: ", end_train_loss, "Training acc: ", end_train_acc,
  "Validation loss:", end_val_loss, "Validation acc:", end_val_acc)
print(decoder)

print('PVAE encoder & SVAE decoder Models Loaded')

# Loss Function
def loss_function(recon_x, x, mu_loss, logvar_loss, alpha=0.5, beta=1):
    MSE = F.mse_loss(recon_x, x.view(-1, INPUT_SIZE))
    KLD = -0.5 * torch.sum(1 + logvar_loss - mu_loss.pow(2) - logvar_loss.exp())
    joint_loss = alpha * MSE + (1 - alpha) * beta * KLD  # From PaccMann # Kingma & Welling. ICLR 2014
    return joint_loss, MSE, KLD


# Train Test Fn
def train_test_run(type_T, data_loader, pvae_encoder, decoder, svae_optimizer):

    if type_T == 'TRAIN':
        pvae_encoder.eval().to(DEVICE)
        decoder.train().to(DEVICE)
    else:
        pvae_encoder.eval().to(DEVICE)
        decoder.eval().to(DEVICE)

    for batch_idx, (smiles, genex) in enumerate(data_loader):
        total_loss = torch.zeros(1, 1).to(DEVICE)
        joint_loss = torch.zeros(1, 1).to(DEVICE)
        batch_loss = torch.zeros(1, 1).to(DEVICE)
        batch_kld_loss = 0.
        batch_total_loss = 0.

        prediction = []
        g_t = []
        decoder_stack = decoder.decoder_stackgru.initStack()
        smiles = smiles.to(DEVICE)
        genex = genex.view(BATCH_SIZE, X_DIM).to(DEVICE)

        input_tensor_smiles, target_tensor_smiles = smiles[:, 0:-1], smiles[:, 1:]
        input_length = input_tensor_smiles.size(1)
        mean, logvar = pvae_encoder.encode(genex.float())
        latent_z = utils.reparameterization_trick(mean, logvar)
        decoder_hidden = decoder.latent_vector_to_hidden(latent_z)  # Initializing the decoder hidden state

        for k in range(input_length): # same length of the input
            output, decoder_hidden, decoder_stack = decoder(input_tensor_smiles[:, k], decoder_hidden, decoder_stack)
            total_loss += criterion(output.view(-1, vocab_size).to(DEVICE), target_tensor_smiles[:, k].view(-1).to(DEVICE))
            for b in range(BATCH_SIZE):
                prediction.append(np.argmax(output[0, b, :].cpu().detach().numpy()))
                g_t.append(target_tensor_smiles[b, k].cpu().detach().numpy())

        kld_loss = utils.kl_divergence_loss(mean, logvar, KL_GROWTH_RATE, EPOCHS + 1, False).to(DEVICE)
        batch_kld_loss += kld_loss.item()
        batch_total_loss += total_loss.item()
        joint_loss += (total_loss + kld_loss)
        batch_loss += joint_loss.item()

        if type_T == 'TRAIN':
            svae_optimizer.zero_grad()
            joint_loss = joint_loss.to(DEVICE)
            joint_loss.backward()
            svae_optimizer.step()

    # Report Loss
    if type_T == 'TRAIN':
        loss_avg  = batch_loss / len(data_loader_train.dataset)
        batch_kld_avg = batch_kld_loss / len(data_loader_train.dataset)
        batch_total_avg = batch_total_loss / len(data_loader_train.dataset)
    else:
        loss_avg = batch_loss / len(data_loader_test.dataset)
        batch_kld_avg = batch_kld_loss / len(data_loader_test.dataset)
        batch_total_avg = batch_total_loss / len(data_loader_test.dataset)

    return loss_avg, batch_kld_avg, batch_total_avg


# Run Epoch
print('Starting the Train & Test process')
for epoch in range(EPOCHS):
    loss_avg, batch_kld_avg, batch_total_avg = train_test_run('TRAIN', data_loader_train, pvae_encoder, decoder, svae_optimizer)
    print(EPOCHS, epoch, 'TRAIN', len(data_loader_train.dataset), loss_avg.item(), batch_kld_avg, batch_total_avg)

    loss_avg, batch_kld_avg, batch_total_avg = train_test_run('TEST', data_loader_test, pvae_encoder, decoder, svae_optimizer)
    print(EPOCHS, epoch, 'TEST', len(data_loader_test.dataset), loss_avg.item(), batch_kld_avg, batch_total_avg)

    if epoch % SAVE_INTERVAL == 0:
        state = {
            'epoch': epoch,
            'state_dict': decoder.state_dict(),
            'optimizer': svae_optimizer.state_dict(),
        }
        savepath = PATH_TO_SAVE + str(epoch) + '_' + model_name + '.pth'
        torch.save(state, savepath)

print('Completed Train & Test')


