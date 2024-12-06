import sys
sys.path.append('/home/svilhes/Bureau/these/AnoControl/mnist/OneVSAll/')

import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import json
import os

from models.fanogan import ConvDiscriminator, ConvGenerator, Encoder
from mnist import get_mnist

DEVICE="cuda"

LR=2e-4
BATCH_SIZE=128
EPOCHS=20
EPOCHS_ENCODER=10
LATENT_DIM=100

beta_1 = 0.5
beta_2 = 0.999

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

for normal in range(10):

    NORMAL=normal
    print(f'anormal digit = {NORMAL}')

    trainset, valset, test_dict_dataset = get_mnist(normal_digit=NORMAL, gcn=False)

    disc = ConvDiscriminator(im_channel=1, hidden_dim=64).to(DEVICE)
    gen = ConvGenerator(z_dim=LATENT_DIM, hidden_dim=64).to(DEVICE)

    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    optim_gen = optim.Adam(gen.parameters(), lr=LR, betas=(beta_1, beta_2))
    optim_disc = optim.Adam(disc.parameters(), lr=LR, betas=(beta_1, beta_2))

    criterion_disc = nn.BCEWithLogitsLoss()
    criterion_gen = nn.BCEWithLogitsLoss()

    pbar = trange(EPOCHS, desc="Training of the GAN")
    for epoch in pbar:
        epoch_loss_gen = 0
        epoch_loss_disc = 0

        for i, inputs in enumerate(trainloader):

            inputs = inputs.to(DEVICE)
            batch_size = inputs.size(0)

            # Discriminator

            optim_disc.zero_grad()

            ones = torch.ones(batch_size, 1).to(DEVICE)

            pred_disc_true = disc(inputs)
            loss_disc_true = criterion_disc(pred_disc_true, ones)


            zeros = torch.zeros(batch_size, 1).to(DEVICE)

            z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            fake = gen(z)
            pred_disc_false = disc(fake.detach())
            loss_disc_fake = criterion_disc(pred_disc_false, zeros)

            loss_disc_true.backward()
            loss_disc_fake.backward()

            loss_disc = (loss_disc_true + loss_disc_fake) /2
            optim_disc.step()

            # Generator :
            optim_gen.zero_grad()

            pred_disc_false = disc(fake)
            loss_gen = criterion_gen(pred_disc_false, ones)

            loss_gen.backward()
            optim_gen.step()

            epoch_loss_gen+=loss_gen.item()
            epoch_loss_disc+=loss_disc_fake.item() + loss_disc_true.item()

    gen.eval()
    disc.eval()

    encoder = Encoder(in_channels=1, hidden_channels=64, z_dim=LATENT_DIM).to(DEVICE)
    optim_encoder = optim.Adam(encoder.parameters(), lr=1e-4)

    f = nn.Sequential(*list(disc.disc.children())[:-1])  # On exclut la dernière couche (Conv2d)
    f.append(nn.Flatten(start_dim=1))
    
    pbar = trange(EPOCHS_ENCODER, desc="Training of the Encoder")   
    for epoch in pbar:
        curr_loss = 0

        for batch in trainloader:
            batch = batch.to(DEVICE)

            encoded = encoder(batch)
            decoded = gen(encoded)

            loss_residual = nn.functional.mse_loss(decoded, batch)

            features_decoded = f(decoded).flatten(start_dim=1)
            features_batch = f(batch).flatten(start_dim=1)

            loss_discriminator = nn.functional.mse_loss(features_decoded, features_batch)

            complete_loss = loss_residual + loss_discriminator

            optim_encoder.zero_grad()
            complete_loss.backward()
            optim_encoder.step()

            curr_loss+=complete_loss.item()
        pbar.set_description(f"For epoch {epoch+1}/{EPOCHS} ; loss : {curr_loss/len(trainloader)}")

    encoder.eval()
    f.eval()

    final_results = {i:None for i in range(10)}

    for i in range(10):
        test_inputs = test_dict_dataset[i].to(DEVICE)
        with torch.no_grad():
            encoded = encoder(test_inputs)
            decoded = gen(encoded)
        loss_residual = nn.functional.mse_loss(decoded, test_inputs)
        features_decoded = f(decoded).flatten(start_dim=1)
        features_batch = f(test_inputs).flatten(start_dim=1)

        loss_discriminator = nn.functional.mse_loss(features_decoded, features_batch)

        complete_loss = loss_residual + loss_discriminator
        final_results[i]=complete_loss.item()

    os.makedirs("results/figures/fanogan", exist_ok=True)

    plt.bar(final_results.keys(), final_results.values())
    plt.title(f'Mean Loss for each digit : NORMAL = {NORMAL}')
    plt.savefig(f'results/figures/fanogan/mean_scores_{NORMAL}.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

    valset = torch.stack([valset[i] for i in range(len(valset))]).to(DEVICE)

    with torch.no_grad():

        encoded = encoder(valset)
        reconstructed = gen(encoded)

        features_decoded = f(reconstructed)
        features_batch = f(valset)

    reconstructed_loss = reconstructed.flatten(start_dim=1)
    normal_val_loss = valset.flatten(start_dim=1).to(DEVICE)
    features_decoded = features_decoded.flatten(start_dim=1)
    features_batch = features_batch.flatten(start_dim=1)

    loss_residual = ((reconstructed_loss - normal_val_loss)**2).sum(dim=1)
    loss_discriminator = ((features_decoded - features_batch)**2).sum(dim=1)

    complete_loss = loss_residual+loss_discriminator

    val_scores = - complete_loss

    val_scores_sorted, indices = val_scores.sort()

    final_results = {i:[None, None] for i in range(10)}

    for digit in range(10):
        inputs_test = test_dict_dataset[digit].to(DEVICE)

        with torch.no_grad():
            encoded = encoder(inputs_test)
            reconstructed = gen(encoded)
            features_decoded = f(reconstructed)
            features_batch = f(inputs_test)

        reconstructed_loss = reconstructed.flatten(start_dim=1)
        inputs_test_loss = inputs_test.flatten(start_dim=1).to(DEVICE)
        features_decoded = features_decoded.flatten(start_dim=1)
        features_batch = features_batch.flatten(start_dim=1)

        loss_residual = ((reconstructed_loss - inputs_test_loss)**2).sum(dim=1)
        loss_discriminator = ((features_decoded - features_batch)**2).sum(dim=1)

        complete_loss = loss_residual+loss_discriminator

        test_scores = - complete_loss

        test_p_values = (1 + torch.sum(test_scores.unsqueeze(1) >= val_scores_sorted, dim=1)) / (len(val_scores_sorted) + 1)

        final_results[digit][0] = test_p_values.tolist()
        final_results[digit][1] = len(inputs_test)

    os.makedirs("results/p_values/fanogan", exist_ok=True)

    with open(f"results/p_values/fanogan/pval_{NORMAL}.json", "w") as file:
        json.dump(final_results, file)