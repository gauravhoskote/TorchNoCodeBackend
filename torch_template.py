import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import json
from mappers import optimMapper, lossMapper, datasetMapper, layerMapper
import torch.nn.functional as F
import os
import logging


LOG_DIRECTORY = 'logs'
PARAM_DIRECTORY = 'model_param_files'
MODEL_DIRECTORY = 'models'

def param_reader(layers):
  layer_list = []
  for layer in layers:
      layer_list.append(layerMapper(layer['layer_type'], layer))
  return layer_list

class NeuralNet(nn.Module):
  def __init__(self, input, layers):
    super(NeuralNet, self).__init__()
    self.input = input
    self.params = param_reader(layers)
    self.layers = nn.ModuleList(self.params)

  #forward pass
  def forward(self, x):
      for i, layer in enumerate(self.layers):
          x = layer(x)
      return x



#Training
def train_model(epochs, train_dataloader, model, optimizer, criterion):
    for i in range(epochs):
        filewriter.write(marker + f'Running Epoch #{i+1}' + '\n')
        for batch , (features, labels) in enumerate(train_dataloader):
            predictions = model(features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    model.eval() #for batch norm and other layers


def get_accuracy(loader, model, criterion):
    num_samples = 0
    num_correct = 0
    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            scores = model(features)
            filewriter.write(marker + scores + '\n')
            filewriter.write(marker + labels + '\n')
            loss = criterion(scores, labels)
            filewriter.write(marker + 'Loss: ' + str(loss) + '\n')
            break



def create_model(id):
    logfile = id+'.log'
    logpath = os.path.join(LOG_DIRECTORY, logfile)
    # logging.basicConfig(filename=logpath, filemode='w', level=logging.DEBUG)
    global filewriter
    global marker
    marker = str(id) + ':'
    filewriter = open(logpath, "a")
    filewriter.write(marker + "Initiating model creation" + '\n')
    filename = id + '.json'
    filepath = os.path.join(PARAM_DIRECTORY, filename)
    json_file = open(filepath)
    param_dict = json.load(json_file)
    layers = param_dict['layers']
    model = NeuralNet(param_dict['input_size'], layers)

    # Dataset and DataLoaders
    if param_dict['preloaded_dataset']:
        training_data, test_data = datasetMapper(param_dict['dataset']['id'])
    else:
        filewriter.write(marker + 'Custom Dataset TBD' + '\n')

    # filewriter.write(marker + model.layers + '\n')
    train_dataloader = DataLoader(training_data, param_dict['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_data, param_dict['batch_size'], shuffle=True)

    # Set Hyperparameters
    criterion = lossMapper(param_dict['loss_function'])
    optimizer = optimMapper(param_dict['optimizer'], param_dict['learning_rate'], model)
    epochs = param_dict['epochs']
    train_model(epochs, train_dataloader, model, optimizer, criterion)
    filewriter.close()
    modelname = id + '.pt'
    modelpath = os.path.join(MODEL_DIRECTORY,modelname)
    torch.save(model, modelpath)
    # get_accuracy(test_dataloader, model, criterion)


# create_model()    run this function to test the script independently
