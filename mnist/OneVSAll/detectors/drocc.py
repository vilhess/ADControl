import sys
sys.path.append('/home/svilhes/Bureau/these/AnoControl/mnist/OneVSAll')

import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import json
import os

from models.drocc import MNIST_LeNet, DROCCTrainer, adjust_learning_rate
from mnist import get_mnist

DEVICE="cuda"

BATCH_SIZE=128
LR=1e-4
EPOCHS=10

ONLY_CE_EPOCHS=0
ASCENT_STEP_SIZE=0.1
ASCENT_NUM_STEPS=50

LAMBDA=1
RADIUS=16
GAMMA=2

for normal in range(10):

    NORMAL=normal
    print(f'anormal digit = {NORMAL}')

    trainset, valset, test_dict_dataset = get_mnist(normal_digit=NORMAL, gcn=False, drocc=True)

    model = MNIST_LeNet().to(DEVICE)

    trainset = TensorDataset(trainset, torch.ones(len(trainset), 1))
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE)

    test_inputs, test_targets = [], []
    for key, val in test_dict_dataset.items():
        size = len(val)
        if key==NORMAL:
            target = torch.ones(size, 1)
        else:
            target = torch.zeros(size, 1)
        test_inputs.append(val)
        test_targets.append(target)

    test_inputs = torch.cat(test_inputs)
    test_targets = torch.cat(test_targets)
    testset = TensorDataset(test_inputs, test_targets)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    trainer = DROCCTrainer(model, optimizer, LAMBDA, RADIUS, GAMMA, DEVICE)

    trainer.train(train_loader, learning_rate=LR, lr_scheduler=adjust_learning_rate, total_epochs=EPOCHS, ascent_step_size=ASCENT_STEP_SIZE, only_ce_epochs = ONLY_CE_EPOCHS)
    model = trainer.model

    model.eval()
    test_results = {i:None for i in range(10)}

    with torch.no_grad():
        for i in range(10):
            inputs = test_dict_dataset[i].to(DEVICE)
            logits = model(inputs)
            score = torch.squeeze(logits).cpu()
            score = torch.mean(score)
            test_results[i]=score

    os.makedirs("results/figures/drocc", exist_ok=True)

    plt.bar(test_results.keys(), test_results.values())
    plt.title(f'Mean Loss for each digit : NORMAL = {NORMAL}')

    plt.savefig(f'results/figures/drocc/mean_scores_{NORMAL}.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

    valset = torch.stack([valset[i] for i in range(len(valset))]).flatten(start_dim=1).to(DEVICE)

    with torch.no_grad():
        val_logits = model(valset)

    final_results = {i:[None, None] for i in range(10)}

    for digit in range(10):

        inputs_test = test_dict_dataset[digit].flatten(start_dim=1).to(DEVICE)
        with torch.no_grad():
            test_logits = model(inputs_test)

        test_p_values = (1 + torch.sum(test_logits >= val_logits.squeeze(1), dim=1)) / (len(val_logits.squeeze(1)) + 1)

        final_results[digit][0] = test_p_values.cpu().tolist()
        final_results[digit][1] = len(inputs_test)

    os.makedirs("results/p_values/drocc", exist_ok=True)

    with open(f"results/p_values/drocc/pval_{NORMAL}.json", "w") as file:
        json.dump(final_results, file)