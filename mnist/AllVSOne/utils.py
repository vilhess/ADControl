import torch 
import numpy as np

def get_dataset_by_digit(dataset):
    dic = {i:[] for i in range(10)}
    for img, lab in dataset:
        dic[lab].append(img)
    dic = {i:torch.stack(dic[i]) for i in range(10)}
    return dic

def global_contrast_normalization(x: torch.tensor, scale='l2'):
    n_features = int(np.prod(x.shape))
    mean = torch.mean(x)
    x -= mean
    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale
    return x