import sys
sys.path.append('/home/svilhes/Bureau/these/projections/OneVSAll')

import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from models.vae import VAE
from losses.loss import LossVAE
from mnist import get_mnist

DEVICE="cuda"

LR=3e-4
BATCH_SIZE=128
EPOCHS=20

labels = {  0:"red",
            1:"blue",
            2: 'green',
            3: 'yellow',
            4: 'purple',
            5: 'orange',
            6: 'pink',
            7: 'brown',
            8: 'gray',
            9: 'cyan'
          }

for normal in range(10):
    
    NORMAL=normal
    print(f'normal digit = {NORMAL}')

    trainset, valset, test_dict_dataset = get_mnist(normal_digit=NORMAL, gcn=False)

    model = VAE(in_dim=784, hidden_dim=[512, 256], latent_dim=2).to(DEVICE)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    criterion = LossVAE()
    pbar = trange(EPOCHS, desc="Training")
    for epoch in pbar:
        epoch_loss=0
        for inputs in trainloader:
            inputs = inputs.to(DEVICE).flatten(start_dim=1)
            reconstructed, mu, logvar = model(inputs)
            loss = criterion(inputs, reconstructed, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
        pbar.set_description(f"For epoch {epoch+1}/{EPOCHS} ; loss : {epoch_loss}")

    model.eval()
    plt.figure(figsize=(12, 12))

    for digit, images in test_dict_dataset.items():
        images = images[:200].to(DEVICE).flatten(start_dim=1)
        with torch.no_grad():
            mu, logvar = model.encoder(images)
        z = model.rep_trick(mu, logvar).cpu()
        if digit==NORMAL:
            plt.scatter(z[:,0], z[:,1], c=labels[digit], label=f'{digit} -- NORMAL')
        else:
            plt.scatter(z[:,0], z[:,1], c=labels[digit], label=digit)
    plt.legend()
    plt.title(f"2D latent space - anormal = {NORMAL}.png")
    plt.savefig(f"figures/vae/latent_space_norm_{NORMAL}.png")
    plt.close