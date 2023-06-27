import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import json


def lossMapper(key):
    mapper_dictionary = {
        'MSE' : nn.MSELoss()
    }
    return mapper_dictionary[key]


def optimMapper(key, lr, model):
    optim_dictionary = {
        'SGD': torch.optim.SGD(model.parameters(), lr = lr)
    }
    return optim_dictionary[key]

def datasetMapper(key):
    if key == 'FashionMNIST':
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )
        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )
        ## Add all datasets here
    return training_data, test_data