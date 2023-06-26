
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import json

json_file = open("./parameters.json")
param_dict = json.load(json_file)

layers = param_dict['layers']
def param_reader():
  layer_list = []
  for layer in layers:
    if layer['layer_type'] == 'linear':
      layer_list.append(nn.Linear(layer['input_size'], layer['output_size'], layer['bias']))
  return layer_list

class NeuralNet(nn.Module):
  def __init__(self, input):
    super(NeuralNet, self).__init__()
    self.input = input
    self.parameters = param_reader()
    #TODO layers to the NN shd be parameters
    # Class cannot be printed otherwise
    self.fc1 = nn.Linear(10, 10)


  #forward pass
  def forward(self, x):
    for layer in self.parameters:
      x = layer(x)
    return x

model = NeuralNet(layers[0]['input_size'])

# Dataset and DataLoaders

if param_dict['preloaded_dataset']:
    if param_dict['dataset']['id'] == 'FashionMNIST':
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


print(model)
# print(model.layers)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


#Training

