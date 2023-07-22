
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import json
from mappers import optimMapper, lossMapper, datasetMapper, layerMapper
import torch.nn.functional as F

# from utils import get_accuracy
json_file = open("./parameters.json")
param_dict = json.load(json_file)
layers = param_dict['layers']

def param_reader():
  layer_list = []
  for layer in layers:
      layer_list.append(layerMapper(layer['layer_type'], layer))
  return layer_list

class NeuralNet(nn.Module):
  def __init__(self, input):
    super(NeuralNet, self).__init__()
    self.input = input
    self.params = param_reader()
    self.layers = nn.ModuleList(self.params)

  #forward pass
  def forward(self, x):
      for i, layer in enumerate(self.layers):
          x = layer(x)
      return x

model = NeuralNet(param_dict['input_size'])
# Dataset and DataLoaders

if param_dict['preloaded_dataset']:
    training_data,test_data = datasetMapper(param_dict['dataset']['id'])
else:
    print('Custom Dataset TBD')


# print(model.layers)
train_dataloader = DataLoader(training_data, param_dict['batch_size'], shuffle=True)
test_dataloader = DataLoader(test_data, param_dict['batch_size'], shuffle=True)

#Set Hyperparameters
criterion = lossMapper(param_dict['loss_function'])
optimizer = optimMapper(param_dict['optimizer'], param_dict['learning_rate'], model)
epochs = param_dict['epochs']


#Training
for i in range(epochs):
    print(f'Running Epoch #{i+1}')
    for batch , (features, labels) in enumerate(train_dataloader):
        predictions = model(features)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
def get_accuracy(loader, model):
    num_samples = 0
    num_correct = 0
    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            scores = model(features)
            print(scores)
            print(labels)
            loss = criterion(scores, labels)
            print('Loss: ' + str(loss))
            break

get_accuracy(test_dataloader, model)
