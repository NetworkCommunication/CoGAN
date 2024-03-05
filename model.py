import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from model.ConvolutionLSTM import ConvLSTM
import torch.optim as optim
import matplotlib.pyplot as plt
# import cv2
import torchvision
from tqdm import tqdm
import data
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

class VAE(nn.Module):
    def __init__(self, T, F, S, hidden_dim, nhead=4):
        super(VAE, self).__init__()
        self.T = T
        self.S = S
        self.F = F

        self.hidden_dim = hidden_dim
        self.latent_dim = hidden_dim
        self.multi_head = nhead

        self.MLP_encoder = nn.Linear(F, hidden_dim)
        self.Q_Linear = nn.Linear(S, S*self.multi_head)
        self.K_Linear = nn.Linear(S, S*self.multi_head)
        self.V_Linear = nn.Linear(S, S*self.multi_head)

        self.LSTM_encoder = nn.Sequential(
            nn.ReLU(),
            LSTMNet(S*hidden_dim, S*hidden_dim, hidden_dim, 2, hidden_dim*2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T*S*F)
        )
    def encode(self, x):
        hidden = self.MLP_encoder(x)
        hidden = hidden.transpose(2, 3)
        query_multi = torch.chunk(self.Q_Linear(hidden), self.multi_head, dim=3)
        key_multi = torch.chunk(self.K_Linear(hidden), self.multi_head, dim=3)
        value_multi = torch.chunk(self.V_Linear(hidden), self.multi_head, dim=3)

        att_hidden_mean = 0
        for (query, key, value) in zip(query_multi, key_multi, value_multi):

            query = query.reshape(-1, query.shape[2], query.shape[-1])
            key = key.reshape(-1, key.shape[2], key.shape[-1])
            value = value.reshape(-1, value.shape[2], value.shape[-1])
            query = query / math.sqrt(self.hidden_dim)
            # Calculate attention score
            attn = torch.bmm(query, key.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            att_hidden = torch.bmm(attn, value)
            att_hidden = att_hidden.reshape(x.shape[0], x.shape[1], -1)
            att_hidden_mean += att_hidden
        att_hidden_mean = att_hidden_mean / self.multi_head
        mu1, logvar1 = torch.chunk(self.LSTM_encoder(att_hidden_mean.float()), 2, dim=1)
        std = torch.exp(0.5 * logvar1)
        eps = torch.randn_like(std)
        z1 = mu1 + eps * std
        return z1, mu1, logvar1

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z1, mu1, logvar1 = self.encode(x)
        KL1 =  - 0.5 * torch.mean(1 + logvar1 - mu1.pow(2) - logvar1.exp())
        self.latent = z1
        x_hat1 = (self.decode(z1)).reshape(-1, self.T, self.S, self.F)
        # Secondary encoding
        with torch.no_grad():
            z2,mu2, logvar2 = self.encode(x_hat1)
        KL2 = - 0.5 * torch.mean(1 + logvar2 - mu2.pow(2) - logvar2.exp())

        return  x_hat1, z1, z2,KL1, KL2
class LSTM(nn.Module):
    def __init__(self, input_size, input_size2, hidden_size, num_layers):

        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.W_ii = nn.Parameter(torch.ones(input_size*2, hidden_size)).to(device)
        self.W_hi = nn.Parameter(torch.ones(hidden_size, hidden_size)).to(device)

        self.W_if = nn.Parameter(torch.ones(input_size*2, hidden_size)).to(device)
        self.W_hf = nn.Parameter(torch.ones(hidden_size, hidden_size)).to(device)

        self.W_ig = nn.Parameter(torch.ones(input_size*2, hidden_size)).to(device)
        self.W_hg = nn.Parameter(torch.ones(hidden_size, hidden_size)).to(device)

        self.W_io = nn.Parameter(torch.ones(input_size*2, hidden_size)).to(device)
        self.W_ho = nn.Parameter(torch.ones(hidden_size, hidden_size)).to(device)
    def forward(self, input, input2, hidden):

        h, c = hidden
        # Splice the hidden state of the input and the previous time step together
        combined = torch.cat((input, h), dim=2)
        # Input gate, forget gate, output gate, and memory state
        i = torch.sigmoid(torch.matmul(combined, self.W_ii)  + torch.matmul(c, self.W_hi))
        f = torch.sigmoid(torch.matmul(combined, self.W_if) + torch.matmul(c, self.W_hf))
        g = torch.tanh(torch.matmul(combined, self.W_ig) + torch.matmul(c, self.W_hg))
        o = torch.sigmoid(torch.matmul(combined, self.W_io) + torch.matmul(c, self.W_ho))
        # Calculate new unit states and hidden states
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c
class LSTMNet(nn.Module):

    def __init__(self, input_size, input_size2, hidden_size, num_layers, output_size):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.lstm = LSTM(input_size, input_size2, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(x.size(0), x.size(1), self.input_size).to(device)
        c0 = torch.zeros(x.size(0), x.size(1), self.hidden_size).to(device)
        out, _ = self.lstm(x, x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(4),
            nn.LeakyReLU(),
            nn.Conv3d(4, 16, kernel_size=2, stride=1, padding=1),  # 32----->64
            # nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 4, kernel_size=1, stride=2, padding=2),  # 64----->128
            # nn.BatchNorm3d(4),
            nn.LeakyReLU(),
            nn.Conv3d(4, 2, kernel_size=3, stride=1, padding=1),  # 64----->128
            # nn.BatchNorm3d(2),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=(1, 7, 5))
        )

    def forward(self, x):
        features = self.main(x)
        classifier1 = self.classifier(features)
        classifier2 = classifier1.view(-1, 1).squeeze(1)

        return classifier1, features



