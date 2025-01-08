import sys
sys.path.append('/home/svilhes/Bureau/these/AnoControl/mnist/OneVSAll/')

import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import trange
import json
import os

from models.vae import Encoder, Decoder
from mnist import get_mnist

DEVICE="cuda"

LR=3e-4
BATCH_SIZE=128
EPOCHS=30
LATENT_DIM=64

for normal in range(10):

    NORMAL=normal
    print(f'normal digit = {NORMAL}')

    trainset, valset, test_dict_dataset = get_mnist(normal_digit=NORMAL, gcn=False)

    enc = Encoder(in_dim=784, hidden_dims=[512, 256], latent_dim=LATENT_DIM).to(DEVICE)
    dec1 = Decoder(latent_dim=LATENT_DIM, hidden_dims=[256, 512], out_dim=784).to(DEVICE)
    dec2 = Decoder().to(DEVICE)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    optim_ae1 = optim.Adam(list(enc.parameters()) + list(dec1.parameters()), lr=LR)
    optim_ae2 = optim.Adam(list(enc.parameters()) + list(dec2.parameters()), lr=LR)

    enc.train()
    dec1.train()
    dec2.train()

    pbar = trange(EPOCHS, desc="Training")
    for epoch in pbar:
        curr_loss_1 = 0
        curr_loss_2 = 0
        for inputs in trainloader:
            inputs = inputs.to(DEVICE)
            inputs = inputs.flatten(start_dim=1)

            z = enc(inputs)

            w1 = dec1(z)
            w2 = dec2(z)
            w3 = dec2(enc(w1))

            loss1 = 1/(epoch+1) * torch.mean((x - w1)**2) + (1 - 1/(epoch+1)) * torch.mean((x - w3)**2)
            optim_ae1.zero_grad()
            loss1.backward()
            optim_ae1.step()

            z = enc(inputs)

            w1 = dec1(z)
            w2 = dec2(z)
            w3 = dec2(enc(w1))

            loss2 = 1/(epoch+1) * torch.mean((x - w2)**2) - (1 - 1/(epoch+1)) * torch.mean((x - w3)**2)
            optim_ae2.zero_grad()
            loss2.backward()
            optim_ae2.step()

            curr_loss_1+=loss1.item()
            curr_loss_2+=loss2.item()

        pbar.set_description(f"For epoch {epoch+1}/{EPOCHS} ; loss1 : {curr_loss_1/len(trainloader)} ; loss2 : {curr_loss_2/len(trainloader)}")

    # checkpoints = {'state_dict':model.state_dict()}
    # torch.save(checkpoints, f'checkpoints/conv2_model_anomaly_{ANORMAL}.pkl')

    enc.eval()
    dec1.eval()
    dec2.eval()

    alpha, beta = .5, .5

    test_results = {i:None for i in range(10)}

    all_scores=[]
    all_labels=[]

    with torch.no_grad():
        for i in range(10):
            inputs = test_dict_dataset[i].to(DEVICE)
            inputs = inputs.flatten(start_dim=1)
            w1 = dec1(enc(inputs))
            w2 = dec2(enc(w1))
            errors = alpha * torch.mean((inputs - w1)**2, dim=1) + beta * torch.mean((inputs - w2)**2, dim=1)
            test_results[i]=errors.mean().item()

            all_scores.append(-errors)
            if i==NORMAL:
                target = torch.ones(len(errors))
            else:
                target = torch.zeros(len(errors))
            all_labels.append(target)
            
    all_scores, all_labels = torch.cat(all_scores).cpu(), torch.cat(all_labels).cpu()
    auc = roc_auc_score(all_labels, all_scores)
    print(auc)

    with open('results/roc_auc.json', 'r') as f:
        results = json.load(f)
    if 'usad' not in results.keys():
        results['usad']={}
    results["usad"][f"Normal_{NORMAL}"] = auc
    with open('results/roc_auc.json', 'w') as f:
        json.dump(results, f)

    os.makedirs("results/figures/usad", exist_ok=True)

    plt.bar(test_results.keys(), test_results.values())
    plt.title(f'Mean Loss for each digit : NORMAL = {NORMAL}')

    plt.savefig(f'results/figures/vae/mean_scores_{NORMAL}.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

    valset = torch.stack([valset[i] for i in range(len(valset))]).flatten(start_dim=1).to(DEVICE)

    with torch.no_grad():
        val_reconstructed, _, _ = model(valset)

    val_scores = -torch.sum(((valset - val_reconstructed)**2).flatten(start_dim=1), dim=1)

    final_results = {i:[None, None] for i in range(10)}

    for digit in range(10):

        inputs_test = test_dict_dataset[digit].flatten(start_dim=1).to(DEVICE)
        with torch.no_grad():
            test_reconstructed, _, _ = model(inputs_test)

        test_scores = -torch.sum(((inputs_test - test_reconstructed)**2).flatten(start_dim=1), dim=1)

        test_p_values = (1 + torch.sum(test_scores.unsqueeze(1) >= val_scores, dim=1)) / (len(val_scores) + 1)

        final_results[digit][0] = test_p_values.tolist()
        final_results[digit][1] = len(inputs_test)

    os.makedirs("results/p_values/vae", exist_ok=True)

    with open(f"results/p_values/vae/pval_{NORMAL}.json", "w") as file:
        json.dump(final_results, file)