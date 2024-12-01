import torch 
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Grayscale, Normalize, Resize
import os
from PIL import Image

transform = Compose([
    ToTensor(),
    Grayscale(),
    Resize((28, 28))
])

class CrackDataset(Dataset):
    def __init__(self, root="../../Datasets/Crack", train=True, transform=transform):
        
        self.transform = transform

        if train:
            self.root = os.path.join(root, "Negative")
        else:
            self.root = os.path.join(root, "Positive")
        self.img_names = os.listdir(self.root)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.root, img_name)
        img = Image.open(img_path)
        img = self.transform(img)
        return img