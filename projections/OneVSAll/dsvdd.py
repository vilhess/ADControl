import sys
sys.path.append('/home/svilhes/Bureau/these/projections/OneVSAll')

import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import trange
import json
import os

from models.deepsvdd import MNIST_LeNet
from mnist import get_mnist

DEVICE="cuda"
OBJECTIVE="ONE"

LR=1e-3
EPOCHS=150
BATCH_SIZE=128
WEIGHT_DECAY=1e-6

colors = {  0:"red",
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
    print(f'anormal digit = {NORMAL}')

    trainset, valset, test_dict_dataset = get_mnist(normal_digit=NORMAL, gcn=True)

    model = MNIST_LeNet().to(DEVICE)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0], gamma=0.1)

    # Center initialization : 
    n_samples = 0
    eps=0.1
    c = torch.zeros(model.rep_dim).to(DEVICE)

    model.eval()
    with torch.no_grad():
        for digits in trainloader:
            digits = digits.to(DEVICE)
            outputs = model(digits)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)
    c /= n_samples

    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    model.train()

    pbar = trange(EPOCHS, desc="Training")
    for epoch in pbar:
            
        curr_loss = 0
        for digits in trainloader:
            digits = digits.to(DEVICE)
            optimizer.zero_grad()
            projects = model(digits)
            dist = torch.sum((projects - c) ** 2, dim=1)
            loss = torch.mean(dist)
            curr_loss+=loss.item()
                
            loss.backward()
            optimizer.step()

        scheduler.step()
        pbar.set_description(f"For epoch {epoch+1}/{EPOCHS} ; loss : {curr_loss/len(trainloader)}")

    labels = []
    preds = []

    model.eval()
    with torch.no_grad():
        for digit in range(10):
            inputs = test_dict_dataset[digit].to(DEVICE)
            preds_batch = model(inputs)
            preds_batch = preds_batch.cpu().tolist()
            labels = labels + [digit]*len(inputs)
            preds = preds + preds_batch
    preds.append(c.detach().cpu().tolist())

    X_embedded = TSNE(n_components=2, learning_rate='auto',

                    init='random', perplexity=3).fit_transform(np.asarray(preds))

    center_embed = X_embedded[-1]
    X_embedded = X_embedded[:-1]

    plt.figure(figsize=(20, 10))

    unique_labels = set(labels)
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]  # Indices of this label
        points = [X_embedded[i] for i in indices]  # Extract points
        x_coords, y_coords = zip(*points)  # Separate x and y coordinates

        legend = label
        if label == NORMAL:
            legend = str(legend)+"-NORMAL"
        
        plt.scatter(
            x_coords, y_coords,
            c=colors.get(label, "black"),  # Default to black if no color mapping
            label=legend
        )

    plt.scatter(center_embed[0], center_embed[1], c='black', marker='x', s=200, linewidths=5, label='Center') 

    plt.title('2D T-SNE Embedding of the latent space of DSVDD')
    plt.legend()
    plt.savefig(f'figures/dsvdd/latent_space_norm_{NORMAL}.png')
    plt.close()