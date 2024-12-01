import torch
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torchvision.transforms import ToTensor, Compose, Lambda, Normalize
from copy import deepcopy

from utils import get_dataset_by_digit, global_contrast_normalization

MIN_MAX = [(-0.8826567065619495, 9.001545489292527),
           (-0.6661464580883915, 20.108062262467364),
           (-0.7820454743183202, 11.665100841080346),
           (-0.7645772083211267, 12.895051191467457),
           (-0.7253923114302238, 12.683235701611533),
           (-0.7698501867861425, 13.103278415430502),
           (-0.778418217980696, 10.457837397569108),
           (-0.7129780970522351, 12.057777597673047),
           (-0.8280402650205075, 10.581538445782988),
           (-0.7369959242164307, 10.697039838804978)]

def get_mnist(root="../../../../coding/Dataset/", download=False, normal_digit=0, gcn=False):

    if gcn:

        min_value = MIN_MAX[normal_digit][0]
        max_value = MIN_MAX[normal_digit][1]

        transorm = Compose([
            ToTensor(),
            Lambda(lambda x: global_contrast_normalization(x, scale="l1")),
            Normalize([min_value], [max_value - min_value])
        ])

    else:
        transorm = ToTensor()

    trainset = MNIST(root=root, train=True, download=download, transform=transorm)
    testset = MNIST(root=root, train=True, download=download, transform=transorm)
    train_dict_dataset = get_dataset_by_digit(trainset)

    normal_train = train_dict_dataset[normal_digit]

    train_size = int(0.8*len(normal_train))
    val_size = len(normal_train) - train_size

    generator = torch.Generator().manual_seed(42)
    normal_train, normal_val = random_split(normal_train, [train_size, val_size], generator=generator)
    normal_train, normal_val = torch.stack([normal_train[i] for i in range(len(normal_train))]), torch.stack([normal_val[i] for i in range(len(normal_val))])

    test_dict_dataset = get_dataset_by_digit(testset)

    return normal_train, normal_val, test_dict_dataset

if __name__=="__main__":
    trainset, valset, test_dict_dataset = get_mnist()
    print(len(trainset), len(valset), test_dict_dataset.keys())