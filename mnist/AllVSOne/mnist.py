import torch
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torchvision.transforms import ToTensor, Compose, Lambda, Normalize
from copy import deepcopy

from utils import get_dataset_by_digit, global_contrast_normalization


def get_mnist(root="/home/svilhes/Bureau/Datasets", download=False, anormal_digit=0, gcn=False, drocc=False):

    trainset = MNIST(root=root, train=True, download=download, transform=ToTensor())
    train_dict_dataset = get_dataset_by_digit(trainset)

    normals = [i for i in range(10)]
    normals.remove(anormal_digit)

    normal_train = torch.cat([train_dict_dataset[i] for i in normals])

    train_size = int(0.8*len(normal_train))
    val_size = len(normal_train) - train_size

    generator = torch.Generator().manual_seed(42)
    normal_train, normal_val = random_split(normal_train, [train_size, val_size], generator=generator)
    normal_train, normal_val = torch.stack([normal_train[i] for i in range(len(normal_train))]), torch.stack([normal_val[i] for i in range(len(normal_val))])

    trainset_normal = deepcopy(normal_train)

    if gcn:
        normalized = global_contrast_normalization(trainset_normal, scale="l1")

        min_value = normalized.min()
        max_value = normalized.max()

        train_transorm = Compose([
            Lambda(lambda x: global_contrast_normalization(x, scale="l1")),
            Normalize([min_value], [max_value - min_value])
        ])

        normal_train = train_transorm(normal_train)
        normal_val = train_transorm(normal_val)

        test_transorm = Compose([
            ToTensor(),
            Lambda(lambda x: global_contrast_normalization(x, scale="l1")),
            Normalize([min_value], [max_value - min_value])
        ])

    elif drocc:

        train_transorm = Compose([
            Normalize([0.1307], [0.3081])
        ])

        normal_train = train_transorm(normal_train)
        normal_val = train_transorm(normal_val)

        test_transorm = Compose([
            ToTensor(),
            Normalize([0.1307], [0.3081])
        ])
    
    else:
        test_transorm = ToTensor()

    testset = MNIST(root=root, train=False, download=download, transform=test_transorm)
    test_dict_dataset = get_dataset_by_digit(testset)

    return normal_train, normal_val, test_dict_dataset

if __name__=="__main__":
    trainset, valset, test_dict_dataset = get_mnist()
    print(len(trainset), len(valset), test_dict_dataset.keys())