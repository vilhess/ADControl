import sys
sys.path.append('/home/svilhes/Bureau/these/AnoControl/mnist/AllVSOne/')

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

from models.vae import VAE
from losses.vae import LossVAE
from mnist import get_mnist

DEVICE="cuda"

LR=3e-4
BATCH_SIZE=128
EPOCHS=10

for anormal in range(10):

    ANORMAL=anormal
    print(f'anormal digit = {ANORMAL}')

    trainset, valset, test_dict_dataset = get_mnist(anormal_digit=ANORMAL, gcn=False)

    model = VAE(in_dim=784, hidden_dim=[512, 256], latent_dim=10).to(DEVICE)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    criterion = LossVAE()
    pbar = trange(EPOCHS, desc="Training")
    for epoch in pbar:
        epoch_loss=0
        for inputs in trainloader:
            inputs = inputs.to(DEVICE)
            inputs = inputs.flatten(start_dim=1)
            reconstructed, mu, logvar = model(inputs)
            loss = criterion(inputs, reconstructed, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
        pbar.set_description(f"For epoch {epoch+1}/{EPOCHS} ; loss : {epoch_loss}")

    # checkpoints = {'state_dict':model.state_dict()}
    # torch.save(checkpoints, f'checkpoints/conv2_model_anomaly_{ANORMAL}.pkl')

    model.eval()
    test_results = {i:None for i in range(10)}

    all_scores=[]
    all_labels=[]

    with torch.no_grad():
        for i in range(10):
            inputs = test_dict_dataset[i].to(DEVICE)
            inputs = inputs.flatten(start_dim=1)
            reconstructed, _, _ = model(inputs)
            test_score = torch.sum(((inputs - reconstructed)**2), dim=1)
            test_results[i]=test_score.mean().item()

            all_scores.append(-test_score)
            if i==ANORMAL:
                target = torch.zeros(len(test_score))
            else:
                target = torch.ones(len(test_score))
            all_labels.append(target)

    all_scores, all_labels = torch.cat(all_scores).cpu(), torch.cat(all_labels).cpu()
    auc = roc_auc_score(all_labels, all_scores)

    with open('results/roc_auc.json', 'r') as f:
        results = json.load(f)
    if 'vae' not in results.keys():
        results['vae']={}
    results["vae"][f"Anormal_{ANORMAL}"] = auc
    with open('results/roc_auc.json', 'w') as f:
        json.dump(results, f)

    os.makedirs("results/figures/vae", exist_ok=True)

    plt.bar(test_results.keys(), test_results.values())
    plt.title(f'Mean Loss for each digit : ANORMAL = {ANORMAL}')

    plt.savefig(f'results/figures/vae/mean_scores_{ANORMAL}.jpg', bbox_inches='tight', pad_inches=0)
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

    with open(f"results/p_values/vae/pval_{ANORMAL}.json", "w") as file:
        json.dump(final_results, file)