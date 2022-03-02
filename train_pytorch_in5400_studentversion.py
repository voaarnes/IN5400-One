

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

#import matplotlib.pyplot as plt

from torch import Tensor

import time
import os
import numpy as np

import PIL.Image
import sklearn.metrics

from typing import Callable, Optional
from RainforestDataset import RainforestDataset
from YourNetwork import SingleNetwork


RDIR = "C:/data/rainforest/"  #'/itf-fi-ml/shared/IN5400/dataforall/mandatory1/'


def train_epoch(model, trainloader, criterion, device, optimizer):

    #TODO model.train() or model.eval()?

    losses = []
    for batch_idx, data in enumerate(trainloader):
        if (batch_idx %100==0) and (batch_idx>=100):
          print('at batchidx',batch_idx)

        # TODO calculate the loss from your minibatch.
        # If you are using the TwoNetworks class you will need to copy the infrared
        # channel before feeding it into your model.

    return np.mean(losses)


def evaluate_meanavgprecision(model, dataloader, criterion, device, numcl):

    #TODO model.train() or model.eval()?

    #curcount = 0
    #accuracy = 0




    concat_pred = np.empty((0, numcl)) #prediction scores for each class. each numpy array is a list of scores. one score per image
    concat_labels = np.empty((0, numcl)) #labels scores for each class. each numpy array is a list of labels. one label per image
    avgprecs=np.zeros(numcl) #average precision for each class
    fnames = [] #filenames as they come out of the dataloader

    with torch.no_grad():
      losses = []
      for batch_idx, data in enumerate(dataloader):
          if (batch_idx%100==0) and (batch_idx>=100):
            print('at val batchindex: ', batch_idx)

          inputs = data['image'].to(device)
          outputs = model(inputs)
          labels = data['label']

          loss = criterion(outputs, labels.to(device))
          losses.append(loss.item())

          # This was an accuracy computation
          # cpuout= outputs.to('cpu')
          # _, preds = torch.max(cpuout, 1)
          # labels = labels.float()
          # corrects = torch.sum(preds == labels.data)
          # accuracy = accuracy*( curcount/ float(curcount+labels.shape[0]) ) + corrects.float()* ( curcount/ float(curcount+labels.shape[0]) )
          # curcount+= labels.shape[0]

          # TODO: collect scores, labels, filenames


    for c in range(numcl):
      avgprecs[c]= "# TODO, nope it is not sklearn.metrics.precision_score"

    return avgprecs, np.mean(losses), concat_labels, concat_pred, fnames


def traineval2_model_nocv(dataloader_train, dataloader_test ,  model ,  criterion, optimizer, scheduler, num_epochs, device, numcl):

  best_measure = 0
  best_epoch =-1

  trainlosses=[]
  testlosses=[]
  testperfs=[]

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)


    avgloss=train_epoch(model,  dataloader_train,  criterion,  device , optimizer )
    trainlosses.append(avgloss)

    if scheduler is not None:
      scheduler.step()

    perfmeasure, testloss,concat_labels, concat_pred, fnames  = evaluate_meanavgprecision(model, dataloader_test, criterion, device, numcl)
    testlosses.append(testloss)
    testperfs.append(perfmeasure)

    print('at epoch: ', epoch,' classwise perfmeasure ', perfmeasure)

    avgperfmeasure = np.mean(perfmeasure)
    print('at epoch: ', epoch,' avgperfmeasure ', avgperfmeasure)

    if avgperfmeasure > best_measure: #higher is better or lower is better?
      bestweights= model.state_dict()
      #TODO track current best performance measure and epoch

      #TODO save your scores

  return best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs


class yourloss(nn.modules.loss._Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        return

    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        #TODO
        m = nn.Sigmoid()
        loss = nn.BCELoss()
        output = loss(m(input_), target)
        return output


def runstuff():
  config = dict()
  config['use_gpu'] = True #True #TODO change this to True for training on the cluster
  config['lr'] = 0.005
  config['batchsize_train'] = 32
  config['batchsize_val'] = 64
  config['maxnumepochs'] = 35
  config['scheduler_stepsize'] = 10
  config['scheduler_factor'] = 0.3

  # This is a dataset property.
  config['numcl'] = 17

  # Data augmentations.
  data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
  }


  # Datasets
  image_datasets={}
  image_datasets['train'] = RainforestDataset(root_dir=RDIR,trvaltest=0, transform=data_transforms['train'])
  image_datasets['val'] = RainforestDataset(root_dir=RDIR,trvaltest=1, transform=data_transforms['val'])

  # Dataloaders
  #TODO use num_workers=1
  dataloaders = {}
  dataloaders['train']= torch.utils.data.DataLoader(image_datasets['train'], batch_size=config['batchsize_train'], num_workers=1)
  dataloaders['val']= torch.utils.data.DataLoader(image_datasets['val'], batch_size=config['batchsize_val'], num_workers=1)

  # Device
  if True == config['use_gpu']:
      device= torch.device('cuda:0')
  else:
      device= torch.device('cpu')

  # Model 1101s
  # TODO create an instance of the network that you want to use.
  # TwoNetworks()
  print("reached")
  resmodel = resnet18(pretrained=True)
  model = SingleNetwork(resmodel)
  model = model.to(device)

  for param in model.parameters():
      print(param)

  lossfct = yourloss()

  #TODO 1101
  # Observe that all parameters are being optimized
  someoptimizer = torch.optim.Adam(model.parameters(), lr=config['learningRate'])

  # Decay LR by a factor of 0.3 every X epochs
  #TODO
  #somelr_scheduler = torch.optim.LinearLR(someoptimizer, start_factor=0.3, total_iters=config['maxnumepochs'])
  lambda1 = lambda epoch: 0.3 ** epoch
  somelr_scheduler = torch.optim.lr_scheduler.LambdaLR(someoptimizer, lr_lambda=lambda1)

  best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs = traineval2_model_nocv(dataloaders['train'], dataloaders['val'] ,  model ,  lossfct, someoptimizer, somelr_scheduler, num_epochs= config['maxnumepochs'], device = device , numcl = config['numcl'] )


if __name__=='__main__':

  runstuff()
