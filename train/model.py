# This file contains the detail of the pretrained models

import torch
import torch.nn as nn
import torch.nn.functional as F

X_DIM = 978
HIDDEN_DIM = 800
HIDDEN_DIM2 = 512
LATENT_DIM = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available:", device)

class Stack_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, stack_depth, stack_width, batch_size, dropout=0.2, n_layers=1, bidir=False):
        super(Stack_GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.stack_depth = stack_depth
        self.stack_width = stack_width
        self.num_layers = n_layers
        self.bidir = bidir
        self.batch_size = batch_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.stack_controls_layer = nn.Linear(in_features=hidden_size, out_features=3)
        self.stack_input_layer = nn.Linear(in_features=hidden_size, out_features=stack_width)
        self.gru = nn.GRU(input_size=hidden_size + stack_width, hidden_size=hidden_size, num_layers=n_layers, bidirectional=bidir, dropout=dropout)

    def forward(self, inp, hidden, stack):
        embedded_input = self.embedding(inp.view(1, -1)).to(device)
        hidden_ = hidden
        hidden_2_stack = hidden_[-1, :, :]
        stack_controls = self.stack_controls_layer(hidden_2_stack)
        stack_controls = F.softmax(stack_controls, dim=-1)
        stack_input = self.stack_input_layer(hidden_2_stack.unsqueeze(0))
        stack_input = torch.tanh(stack_input)
        stack = self.stack_augmentation(stack_input.permute(1, 0, 2), stack, stack_controls)
        stack_top = stack[:, 0, :].unsqueeze(0)
        inp = torch.cat((embedded_input, stack_top), dim=2)
        output, hidden = self.gru(inp, hidden)
        return output, hidden, stack

    def stack_augmentation(self, input_val, prev_stack, controls):
        batch_size = prev_stack.size(0)
        controls = controls.view(-1, 3, 1, 1)
        zeros_at_the_bottom = torch.zeros(batch_size, 1, self.stack_width, device=device)
        a_push, a_pop, a_no_op = (controls[:, 0], controls[:, 1], controls[:, 2])
        stack_down = torch.cat((prev_stack[:, 1:], zeros_at_the_bottom), dim=1)
        stack_up = torch.cat((input_val, prev_stack[:, :-1]), dim=1)
        new_stack = a_no_op * prev_stack + a_push * stack_up + a_pop * stack_down
        return new_stack

    def initHidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=device)

    def initStack(self):
        return torch.zeros(self.batch_size, self.stack_depth, self.stack_width, device=device)


class VAE_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, z_dim, output_size, batch_size, stack_depth, stack_width, dropout=0.2,
                 n_layers=1, bidir=False):
        super(VAE_Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.stack_depth = stack_depth
        self.stack_width = stack_width
        self.bidir = bidir
        self.batch_size = batch_size
        self.numdir = 1
        self.forward_stackgru = Stack_GRU(input_size, hidden_size, output_size, stack_depth, stack_width, batch_size,
                                          dropout=dropout, n_layers=n_layers, bidir=False)

        self.fine_tune()

        if (self.bidir):
            self.numdir = 2
            self.backward_stackgru = Stack_GRU(input_size, hidden_size, output_size, stack_depth, stack_width,
                                               batch_size, dropout=dropout, n_layers=n_layers, bidir=False)

        self.mean = nn.Linear(self.numdir * hidden_size, z_dim)
        self.logvar = nn.Linear(self.numdir * hidden_size, z_dim)

    def fine_tune(self, fine_tune=False):  ## Defined by Dibya to stop backprop in encoder
        for p in self.forward_stackgru.parameters():
            p.requires_grad = False

    # Forward function for unidirectional Stack-GRU layers in the encoder
    def forward_unidir(self, inp, hidden_forward, stack_forward):
        output, hidden_forward, stack_forward = self.forward_stackgru(inp, hidden_forward, stack_forward)
        return hidden_forward

    # Forward function for bidirectional Stack-GRU layers in the encoder
    def forward_bidir(self, inp1, inp2, hidden_forward, stack_forward, hidden_backward, stack_backward):
        output, hidden_forward, stack_forward = self.forward_stackgru(inp1, hidden_forward, stack_forward)
        output_backward, hidden_backward, stack_backward = self.backward_stackgru(inp2, hidden_backward, stack_backward)
        return hidden_forward, hidden_backward

    def post_gru_reshape_function_unidir(self, hidden):
        hidden = hidden.view(self.num_layers, self.batch_size, self.hidden_size)
        hidden_new = hidden[-1, :, :]
        mu = self.mean(hidden_new)
        sigma = self.logvar(hidden_new)
        return mu, sigma

    def post_gru_reshape_function_bidir(self, hidden1, hidden2):
        hidden1 = hidden1.view(self.num_layers, self.batch_size, self.hidden_size)
        hidden_new_1 = hidden1[-1, :, :]
        hidden2 = hidden2.view(self.num_layers, self.batch_size, self.hidden_size)
        hidden_new_2 = hidden2[-1, :, :]
        # Concatenate the forward and backward hidden states of the last GRU layer of the encoder
        hidden_new = torch.cat([hidden_new_1, hidden_new_2], dim=1)
        mu = self.mean(hidden_new)
        sigma = self.logvar(hidden_new)
        return mu, sigma

    def load_model_weights(self, path):
        weights = torch.load(path)
        self.load_state_dict(weights)

    def save_model_weights(self, path):
        torch.save(self.state_dict(), path)


class VAE_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, z_dim, output_size, batch_size, stack_depth, stack_width, dropout=0.2,
                 n_layers=1, bidir=False):
        super(VAE_Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.num_layers = n_layers
        self.stack_depth = stack_depth
        self.stack_width = stack_width
        self.bidir = bidir
        self.numdir = 1

        self.latent_to_hidden = nn.Linear(z_dim, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.decoder_stackgru = Stack_GRU(input_size, hidden_size, output_size, stack_depth, stack_width, batch_size,
                                          dropout=dropout, n_layers=n_layers,
                                          bidir=False)  # By default the decoder is always unidirectional - Bidirectional decoder has not yet been implemented

    def forward(self, inp, hidden, stack):
        output, hidden, stack = self.decoder_stackgru(inp, hidden, stack)
        output = self.out(output)
        return output, hidden, stack

    def latent_vector_to_hidden(self, latent_z):
        latent_z = latent_z.repeat(self.num_layers, 1, 1)
        hidden = self.latent_to_hidden(latent_z)
        return hidden

    def load_model_weights(self, path):
        weights = torch.load(path)
        self.load_state_dict(weights)

    def save_model_weights(self, path):
        torch.save(self.state_dict(), path)


class VAE(nn.Module):
    def __init__(self, INPUT_SIZE, LATENT_SIZE, OUTPUT_SIZE):
        super(VAE, self).__init__()
        self.INPUT_SIZE = INPUT_SIZE
        self.OUTPUT_SIZE = OUTPUT_SIZE
        self.LATENT_SIZE = LATENT_SIZE
        self.HIDDEN_SIZE_1 = 768
        self.HIDDEN_SIZE_2 = 512
        self.fc1 = nn.Linear(self.INPUT_SIZE, self.HIDDEN_SIZE_1)
        self.fc12 = nn.Linear(self.HIDDEN_SIZE_1, self.HIDDEN_SIZE_2)
        self.fc21 = nn.Linear(self.HIDDEN_SIZE_2, self.LATENT_SIZE)
        self.fc22 = nn.Linear(self.HIDDEN_SIZE_2, self.LATENT_SIZE)
        self.fc41 = nn.Linear(self.LATENT_SIZE, self.HIDDEN_SIZE_2)
        self.fc42 = nn.Linear(self.HIDDEN_SIZE_2, self.HIDDEN_SIZE_1)
        self.fc4 = nn.Linear(self.HIDDEN_SIZE_1, self.OUTPUT_SIZE)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc12(h1))
        mu_enc = self.fc21(h2)
        logvar_enc = self.fc22(h2)
        return mu_enc, logvar_enc

    def reparameterize(self, mu_rep, logvar_rep):
        if self.training:
            std = torch.exp(0.5 * logvar_rep)
            eps = torch.randn_like(std)
            z = mu_rep + eps * std
            return z
        else:
            return mu_rep

    def decode(self, z):
        h1 = F.relu(self.fc41(z))
        h2 = F.relu(self.fc42(h1))
        final = torch.tanh(self.fc4(h2))
        return final

    def forward(self, x):
        mu_fwd, logvar_fwd = self.encode(x.view(-1, self.INPUT_SIZE))
        z = self.reparameterize(mu_fwd, logvar_fwd)
        out = self.decode(z)
        return out, mu_fwd, logvar_fwd

def main():
    print("Hello World")


if __name__ == "__main__":
    main()
else:
    print('Just Imported')
`
