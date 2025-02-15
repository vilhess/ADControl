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

from models.deepsvdd import MNIST_LeNet
from mnist import get_mnist

DEVICE="cuda"
OBJECTIVE="ONE"

LR=1e-3
EPOCHS=150
BATCH_SIZE=128
WEIGHT_DECAY=1e-6

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
            
    # checkpoints = {'state_dict':model.state_dict(),
    #             'center':c}

    # torch.save(checkpoints, f'checkpoints/{OBJECTIVE}/model_anormal_{ANORMAL}.pkl')

    model.eval()
    test_results = {i:None for i in range(10)}

    all_scores=[]
    all_labels=[]

    with torch.no_grad():
        for digit in range(10):
            test_inputs = test_dict_dataset[digit].to(DEVICE)
            with torch.no_grad():
                projs = model(test_inputs)
            dist = torch.sum((projs - c) ** 2, dim=1)
            loss = torch.mean(dist)

            test_results[digit]=loss.item()

            all_scores.append(-dist)
            if digit==NORMAL:
                target = torch.ones(len(dist))
            else:
                target = torch.zeros(len(dist))
            all_labels.append(target)

    all_scores, all_labels = torch.cat(all_scores).cpu(), torch.cat(all_labels).cpu()
    auc = roc_auc_score(all_labels, all_scores)

    with open('results/roc_auc.json', 'r') as f:
        results = json.load(f)
    if 'dsvdd' not in results.keys():
        results['dsvdd']={}
    results["dsvdd"][f"Normal_{NORMAL}"] = auc
    with open('results/roc_auc.json', 'w') as f:
        json.dump(results, f)

    os.makedirs("results/figures/deepsvdd", exist_ok=True)

    plt.bar(test_results.keys(), test_results.values())
    plt.title(f'Mean Loss for each digit : NORMAL = {NORMAL}')
    plt.savefig(f'results/figures/deepsvdd/mean_scores_{NORMAL}.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

    valset = torch.stack([valset[i] for i in range(len(valset))]).to(DEVICE)

    with torch.no_grad():
        projs = model(valset)
    dist = torch.sum((projs - c) ** 2, dim=1)

    scores_val = - dist

    final_results = {i:[None, None] for i in range(10)}

    for digit in range(10):
        digits = test_dict_dataset[digit].to(DEVICE)

        with torch.no_grad():
            projs = model(digits)

        dist = torch.sum((projs - c) ** 2, dim=1)
        scores_test = - dist    
        
        test_p_values = (1 + torch.sum(scores_test.unsqueeze(1) >= scores_val, dim=1)) / (len(scores_val) + 1)
        final_results[digit][0] = test_p_values.tolist()
        final_results[digit][1] = len(digits)

    os.makedirs("results/p_values/deepsvdd", exist_ok=True)

    with open(f"results/p_values/deepsvdd/pval_{NORMAL}.json", "w") as file:
        json.dump(final_results, file)