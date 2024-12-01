import sys 
sys.path.append('../')

import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm

from crack.dataset import CrackDataset
from crack.models.vae import CVAE
from crack.loss.loss_vae import LossVAE

DEVICE="cuda"
BATCH_SIZE=128
EPOCHS=20
LR=3e-4

THRESHOLD = 0.001

normal = CrackDataset(train=True)
anormal = CrackDataset(train=False)

model = CVAE(in_channels=1, hidden_channels=64, latent_dim=10).to(DEVICE)
criterion = LossVAE()
optimizer = optim.Adam(model.parameters(), lr=LR)

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

model.train()

for epoch in range(EPOCHS):
    epoch_loss=0
    model.train()
    for images in tqdm(trainloader):
        images = images.to(DEVICE)
        reconstructed, mu, logvar = model(images)
        loss = criterion(images, reconstructed, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
    print(f"epoch: {epoch+1}/{EPOCHS} ;  loss: {epoch_loss/len(trainloader)}")

model.eval()
with torch.no_grad():
    normal_reconstructed, mu, logvar = model(normal_batch.to(DEVICE))
    anormal_reconstructed, mu, logvar = model(anormal_batch.to(DEVICE))

error_map_normal = torch.abs(normal_reconstructed.cpu() - normal_batch)
error_map_anormal = torch.abs(anormal_reconstructed.cpu() - anormal_batch)

val_errors = []

model.eval()
with torch.no_grad():
    for val_batch in val_loader:
        val_batch = val_batch.to(DEVICE)
        batch_rec, _, _  = model(val_batch)
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

plt.savefig(f"figures/vae.png")

checkpoint = {"state_dict":model.state_dict()}
torch.save(checkpoint, f"checkpoints/vae.pkl")