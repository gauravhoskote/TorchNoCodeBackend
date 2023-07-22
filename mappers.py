import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import json

def layerMapper(key, layer):
    return getattr(nn, key)(**layer['params'])
    #TODO Add Try Catch Block here

def lossMapper(key):
    return getattr(nn, key)()


def optimMapper(key, lr, model):
    return getattr(torch.optim, key)(model.parameters(), lr=lr)

def datasetMapper(key):
    training_data = getattr(datasets, key)(root="data", train=True, download=True, transform=ToTensor())
    test_data = getattr(datasets, key)(root="data", train=False, download=True, transform=ToTensor())
    return training_data, test_data