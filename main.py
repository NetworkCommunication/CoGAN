
import csv
import os
import tqdm
import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.utils.data import DataLoader
import torchvision.utils as vutils


import os

import numpy as np
import torchvision
import tqdm
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.autograd as autograd
import torch
import torch.nn as nn
import torch.functional as F
from tqdm import tqdm
import numpy as np
import copy
import seaborn as sns
from matplotlib.pylab import mpl

from model.model import VAE, Discriminator

import tqdm
from utils.data import load_data
import utils.data

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
mpl.rcParams['figure.figsize'] = 15, 8

import torch
import numpy as np
import os
import re

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


import matplotlib.pyplot as plt
import numpy as np
import os

learning_rate = 0.0001
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)  # Initialize in the BN layer γ， Following a normal distribution of (1, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

def loss_function(mu, logvar):
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_divergence
## Gradient penalty
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1, 1)).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    d_interpolates, _ = D(interpolates)
    fake = torch.ones(*d_interpolates.shape, device=device)
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
lambda_gp = 1
# L1 Loss
def l1_loss(input, target):
    return torch.mean(torch.abs(input - target))
## L2 Loss
def l2_loss(input, target, size_average=True):
    if size_average:
        return torch.mean(torch.pow((input - target), 2))
    else:
        return torch.pow((input - target), 2)
L_adv = nn.MSELoss()
L_con = nn.MSELoss()
L_enc = nn.L1Loss()
PRE_EPOCH = 1
L_bce = nn.BCELoss()

def train(dataLoader, dataloader_test0,dataloader_test3,dataloader_test6,
          dataloader_test9,dataloader_test12,dataloader_test15,
          gen, optimizer, disc_optimizer, epochs, device, scheduler_g, scheduler_d):
    epochs = 100
    history = dict(train=[], train_L_adv_epoch_loss=[])

    for epoch in range(epochs):
        train_losses = []
        train_l_sum, n = 0.0, 0
        val_losses = []
        G_L_adv_epoch_loss = []
        L_adv_epoch_loss = 0.0
        L_total_epoch_loss = 0.0
        disc_epoch_loss = 0.0

        for inputs in tqdm.tqdm(dataLoader):
            inputs = inputs.to(device)  #
            inputs = inputs.permute(0, 2, 1, 3, 4)

            disc_optimizer.zero_grad()
            ''' Update discriminator '''
            D_real, _ = disc(inputs)

            inputs1 = inputs.squeeze(1)
            x_hat1, z1, z2, _, _ = G(inputs1)
            x_hat1 = x_hat1.unsqueeze(1)
            D_fake, _ = disc(x_hat1.detach())

            gradient_penalty = compute_gradient_penalty(disc, inputs1.unsqueeze(1), x_hat1.detach(), device)
            d_loss = (-torch.mean(D_real) + torch.mean(D_fake)
                      ) + gradient_penalty * lambda_gp

            d_loss.backward()
            # a = torch.nn.utils.clip_grad_norm_(disc.parameters(), 10)
            disc_optimizer.step()
            # clip D weights between -0.01, 0.01  Weight clipping
            # for p in disc.parameters():
            #     p.data.clamp_(-0.01, 0.01)
            val_losses.append(d_loss.item())
            disc_epoch_loss += d_loss.item() * batch_size  #

            if i % PRE_EPOCH == 0:
                gen_optimizer.zero_grad()
                outputs1, z3, z4, KL1, KL2 = G(inputs1)
                outputs1= outputs1.unsqueeze(1)
                G_SCORE, _ = disc(outputs1)
                adv_loss = - torch.mean(G_SCORE)
                con_loss = L_con(outputs1, inputs1.unsqueeze(1)) + (KL1 + KL2)  #
                enc_loss = L_enc(z3, z4)
                total_loss = (1 * adv_loss) + \
                             1 * con_loss + \
                             0.1 * enc_loss
                total_loss.backward()
                a = torch.nn.utils.clip_grad_norm_(G.parameters(), 10)
                gen_optimizer.step()
                train_losses.append(total_loss.item())
                G_L_adv_epoch_loss.append(adv_loss.item())

        train_loss = np.mean(train_losses) if i % PRE_EPOCH == 0 else 0
        val_loss = np.mean(val_losses)
        G_L_adv = np.mean(G_L_adv_epoch_loss) if i % PRE_EPOCH == 0 else 0


        scheduler_g.step()
        scheduler_d.step()

        # test(dataloader_test0, dataloader_test3, dataloader_test6,dataloader_test9, dataloader_test12, dataloader_test15, gen)

    return gen, history, disc

def main(G, disc):
    G.train()
    disc.train()

    G, history, disc = train(dataLoader["train"], dataLoader["test0"], dataLoader["test3"], dataLoader["test6"],
                             dataLoader["test9"], dataLoader["test12"], dataLoader["test15"],
                             G, gen_optimizer, disc_optimizer, 10, device,
                             scheduler_g,
                             scheduler_d)


# def test(dataLoader0,dataLoader3,dataLoader6, dataLoader9,dataLoader12,dataLoader15,G):
#     dataLoaders = [dataLoader0, dataLoader3, dataLoader6, dataLoader9, dataLoader12, dataLoader15]
#     reconstructed_data_path = "./data/matrix_data/reconstructed_data/"
#     reconstructed_images_list = []
#     G.train()
#     with torch.no_grad():
#         for dataLoader in dataLoaders:
#             i = 0
#             average_con = 0
#             for x in dataLoader:
#                 i += 1
#                 x = x.to(device)
#                 x = x.permute(0, 2, 1, 3, 4)
#                 x1 = x.squeeze(1)
#                 outputs2, z3, z4, KL1, KL2 = G(x1)
#                 outputs2 = outputs2.unsqueeze(1)
#                 original_images = x1.unsqueeze(1)
#                 reconstructed_images = outputs2
#                 reconstructed_images_list.append(
#                     reconstructed_images.cpu().detach().numpy())
#                 reconstructed_images_array = np.concatenate(reconstructed_images_list, axis=0)
#                 np.save(os.path.join(reconstructed_data_path, "reconstructed_images.npy"), reconstructed_images_array)
#                 average_con += torch.sqrt(torch.mean((reconstructed_images - original_images) ** 2))
#             print(f" {dataLoader}:The difference between the input data and the reconstructed data is {average_con / i}")



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataLoader = load_data()
    G = VAE().to(device)
    disc = Discriminator().to(device)
    G.apply(weights_init)
    disc.apply(weights_init)
    gen_optimizer = torch.optim.Adam(G.parameters(), lr=0.00003)
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=0.00003)
    scheduler_g = MultiStepLR(gen_optimizer, milestones=[ 100, 150, 200, 250,300], gamma=0.5)
    scheduler_d = MultiStepLR(disc_optimizer, milestones=[100, 150, 200, 250,300], gamma=0.5)
    main(G, disc)
