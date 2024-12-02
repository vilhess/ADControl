import sys 
sys.path.append('../')

import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from crack.dataset import CrackDataset
from crack.models.fanogan import ConvGenerator, ConvDiscriminator, Encoder

DEVICE="cuda"

BATCH_SIZE=128
EPOCHS=20
EPOCHS_ENCODER=10
LATENT_DIM=100
LR=2e-4

beta_1 = 0.5
beta_2 = 0.999


THRESHOLD = 0.001

normal = CrackDataset(train=True)
anormal = CrackDataset(train=False)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

disc = ConvDiscriminator(im_channel=1, hidden_dim=64).to(DEVICE)
gen = ConvGenerator(z_dim=LATENT_DIM, hidden_dim=64).to(DEVICE)

gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

optim_gen = optim.Adam(gen.parameters(), lr=LR, betas=(beta_1, beta_2))
optim_disc = optim.Adam(disc.parameters(), lr=LR, betas=(beta_1, beta_2))

criterion_disc = nn.BCELoss()
criterion_gen = nn.BCELoss()

normal_indices = range(len(normal))
train_sizes = int(0.98*len(normal))
train_indices = normal_indices[:train_sizes]
val_indices = normal_indices[train_sizes:-16]
normal_test_indices = normal_indices[-16:]

trainset = Subset(normal, train_indices)
valset = Subset(normal, val_indices)

normal_batch = torch.stack([normal[i] for i in normal_test_indices])
anormal_batch = torch.stack([anormal[i] for i in range(16)])

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)

gen.train()
disc.train()

for epoch in range(EPOCHS):
    epoch_loss_disc=0
    epoch_loss_gen=0
    for images in tqdm(trainloader):

        images = images.to(DEVICE)
        batch_size = images.size(0)

        optim_disc.zero_grad()

        ones = torch.ones(batch_size, 1).to(DEVICE)

        pred_disc_true = disc(images)
        loss_disc_true = criterion_disc(pred_disc_true, ones)

        zeros = torch.zeros(batch_size, 1).to(DEVICE)
        
        z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
        fake = gen(z)

        pred_disc_false = disc(fake.detach())
        loss_disc_fake = criterion_disc(pred_disc_false, zeros)

        loss_disc_fake.backward()
        loss_disc_true.backward()

        loss_disc = (loss_disc_true + loss_disc_fake) / 2
        optim_disc.step()

        optim_gen.zero_grad()
        pred_disc_false = disc(fake)
        loss_gen = criterion_gen(pred_disc_false, ones)
        loss_gen.backward()
        optim_gen.step()

        epoch_loss_disc+=loss_disc.item()
        epoch_loss_gen+=loss_gen.item()

    print(f"epoch: {epoch+1}/{EPOCHS} ;  loss disc: {(epoch_loss_disc)/len(trainloader)} ; loss gen : {(epoch_loss_gen)/len(trainloader)}")


gen.eval()
disc.eval()

encoder = Encoder(in_channels=1, hidden_channels=64, z_dim=LATENT_DIM).to(DEVICE)
optim_encoder = optim.Adam(encoder.parameters(), lr=3e-4)

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
    pbar.set_description(f"For epoch {epoch+1}/{EPOCHS_ENCODER} ; loss : {curr_loss/len(trainloader)}")

encoder.eval()
f.eval()

with torch.no_grad():

    encoded_normal = encoder(normal_batch.to(DEVICE))
    normal_reconstructed = gen(encoded_normal)

    encoded_anormal = encoder(anormal_batch.to(DEVICE))
    anormal_reconstructed = gen(encoded_anormal)

error_map_normal = torch.abs(normal_reconstructed.cpu() - normal_batch)
error_map_anormal = torch.abs(anormal_reconstructed.cpu() - anormal_batch)

val_errors = []

with torch.no_grad():
    for val_batch in val_loader:
        val_batch = val_batch.to(DEVICE)
        val_encoded = encoder(val_batch)
        batch_rec = gen(val_encoded)
        error_map_val = torch.abs(val_batch - batch_rec)
        val_errors.append(error_map_val)
val_errors = torch.cat(val_errors)
val_scores = - val_errors

val_scores_flatten = val_scores.flatten()

seg_normal = []

for normal_score in error_map_normal:
    score = - normal_score
    score = score.flatten()
    pval = (1 + torch.sum(score.unsqueeze(1) >= val_scores_flatten.cpu(), dim=1)) / (len(val_scores_flatten.cpu()) + 1)
    
    error_map = (pval<THRESHOLD).float()
    error_map = error_map.reshape(1, 28, 28)
    seg_normal.append(error_map)
seg_normal = torch.stack(seg_normal)

seg_anormal = []

for anormal_score in error_map_anormal:
    score = - anormal_score
    score = score.flatten()
    pval = (1 + torch.sum(score.unsqueeze(1) >= val_scores_flatten.cpu(), dim=1)) / (len(val_scores_flatten.cpu()) + 1)
    
    error_map = (pval<THRESHOLD).float()
    error_map = error_map.reshape(1, 28, 28)
    seg_anormal.append(error_map)
seg_anormal = torch.stack(seg_anormal)


plt.figure(figsize=(15,10))

plt.subplot(2, 4, 1)

normal_grid = make_grid(normal_batch, nrow=4)
plt.imshow(normal_grid.permute(1, 2, 0), cmap="gray")
plt.axis('off')
plt.title('normal')

plt.subplot(2, 4, 2)

reconstructed_grid = make_grid(normal_reconstructed, nrow=4)
plt.imshow(reconstructed_grid.permute(1, 2, 0).cpu(), cmap="gray")
plt.axis('off')
plt.title('normal reconstructed')

plt.subplot(2, 4, 3)

error_grid = make_grid(error_map_normal, nrow=4)
plt.imshow(error_grid.permute(1, 2, 0), cmap="gray")
plt.axis('off')
plt.title('Normal Error maps')

plt.subplot(2, 4, 4)

grid = make_grid(seg_normal, nrow=4)
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.title(f"Segmentation map: threshold = 0.1%")


plt.subplot(2, 4, 5)

anormal_grid = make_grid(anormal_batch, nrow=4)
plt.imshow(anormal_grid.permute(1, 2, 0), cmap="gray")
plt.axis('off')
plt.title('Anormal')

plt.subplot(2, 4, 6)

reconstructed_grid = make_grid(anormal_reconstructed, nrow=4)
plt.imshow(reconstructed_grid.permute(1, 2, 0).cpu(), cmap="gray")
plt.axis('off')
plt.title('Anormal reconstructed')

plt.subplot(2, 4, 7)

error_grid = make_grid(error_map_anormal, nrow=4)
plt.imshow(error_grid.permute(1, 2, 0), cmap="gray")
plt.axis('off')
plt.title('Anormal Error maps')

plt.subplot(2, 4, 8)

grid = make_grid(seg_anormal, nrow=4)
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.title(f"Segmentation map: threshold = 0.1%")

plt.savefig(f"figures/fanogan.png")

#checkpoint = {"state_dict":model.state_dict()}
#torch.save(checkpoint, f"checkpoints/fanogan.pkl")