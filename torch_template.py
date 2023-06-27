
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import json
from mappers import optimMapper, lossMapper, datasetMapper

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
    self.params = param_reader()
    #TODO layers to the NN shd be parameters
    # Class cannot be printed otherwise
    self.layers = nn.ModuleList(self.params)


  #forward pass
  def forward(self, x):
      x = torch.flatten(x,1)
      for layer in self.layers:
          x = layer(x)
      return x

model = NeuralNet(layers[0]['input_size'])
for param in model.parameters():
    print(param.shape)
# Dataset and DataLoaders

if param_dict['preloaded_dataset']:
    training_data,test_data = datasetMapper(param_dict['dataset']['id'])
else:
    print('Custom Dataset TBD')


print(model)
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
    for i , (features, labels)  in enumerate(train_dataloader):
        predictions = model(features)
        loss = criterion(predictions, labels.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()