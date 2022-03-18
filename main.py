

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

import os
import sys
import math
import time
import zipfile
import numpy as np

from PIL import Image
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from typing import Callable, Optional
from RainforestDataset import RainforestDataset
from RainforestDataset import ChannelSelect
from RainforestDataset import get_classes_list
from YourNetwork import SingleNetwork, TwoNetworks

np.seterr(invalid='ignore')

# Change this to path of dataset
# "train-tif-v2/" Not in variable because csv is in folder above
RDIR = "C:/data/rainforest/"  #'/itf-fi-ml/shared/IN5400/dataforall/mandatory1/'





def train_epoch(model, trainloader, criterion, device, optimizer):
    #TODO model.train() or model.eval()?
    model.train()

    losses = []
    for batch_idx, data in enumerate(trainloader):
        if (batch_idx %100==0) and (batch_idx>=100):
          print('at batchidx',batch_idx)

        # TODO calculate the loss from your minibatch.
        # If you are using the TwoNetworks class you will need to copy the infrared
        # channel before feeding it into your model.

        inputs=data['image'].to(device)
        inputs2=data['irimage'].to(device)
        labels=data['label'].to(device)
        labels = labels.to(torch.float32)
        nuInput = torch.cat((inputs, inputs2), 1)
        modelParam = {}

        if(sys.argv[1]=='1'):
            modelParam = {"inputs":inputs}
        elif(sys.argv[1]=='3'):
            modelParam = {"inputs1":inputs, "inputs2":inputs2}
        elif(sys.argv[1]=='4'):
            modelParam = {"inputs": nuInput}


        optimizer.zero_grad()

        output = model(**modelParam)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if batch_idx%100==0:
          print('current mean of losses ',np.mean(losses))

    return np.mean(losses)





def evaluate_meanavgprecision(model, dataloader, criterion, device, numcl):

    #TODO model.train() or model.eval()?
    model.eval()

    #curcount = 0
    #accuracy = 0


    concat_pred = np.empty((0, numcl))   #prediction scores for each class. each numpy array is a list of scores. one score per image
    concat_labels = np.empty((0, numcl)) #labels scores for each class. each numpy array is a list of labels. one label per image
    avgprecs=np.zeros(numcl)             #average precision for each class
    fnames = []                          #filenames as they come out of the dataloader



    with torch.no_grad():
      losses = []
      for batch_idx, data in enumerate(dataloader):
          if (batch_idx%100==0) and (batch_idx>=100):
            print('at val batchindex: ', batch_idx)

          inputs = data['image'].to(device)
          inputs2 = data['irimage'].to(device)
          labels=data['label']
          labels = labels.to(torch.float32)
          nuInput = torch.cat((inputs, inputs2), 1)
          modelParam = {}

          if(sys.argv[1]=='1' or sys.argv[1]=='5'):
              modelParam = {"inputs":inputs}
          elif(sys.argv[1]=='3'):
              modelParam = {"inputs1":inputs, "inputs2":inputs2}
          elif(sys.argv[1]=='4'):
              modelParam = {"inputs": nuInput}

          outputs = model(**modelParam)
          loss = criterion(outputs, labels.to(device))
          losses.append(loss.item())

          # This was an accuracy computation
          cpuout = outputs.to('cpu')
          #_, preds = torch.max(cpuout, 1)
          # labels = labels.float()
          # corrects = torch.sum(preds == labels.data)
          # accuracy = accuracy*( curcount/ float(curcount+labels.shape[0]) ) + corrects.float()* ( curcount/ float(curcount+labels.shape[0]) )
          # curcount+= labels.shape[0]


          # TODO: collect scores, labels, filenames
          for pred in cpuout:
              temp = np.array(pred)
              concat_pred = np.concatenate((concat_pred, [temp]))
          for label in labels:
              temp = np.array(label)
              temp = temp.astype(int)
              concat_labels = np.concatenate((concat_labels, [temp]))
          for file in data['filename']:
              fnames.append(file)



    for c in range(numcl):
        #avgprecs[c]= "# TODO, nope it is not sklearn.metrics.precision_score"
        val_true, val_pred = concat_labels[:,c].reshape((-1)), concat_pred[:,c].reshape((-1))
        tempAvgprecs = sklearn.metrics.average_precision_score(val_true, val_pred)
        if(np.isnan(tempAvgprecs)): avgprecs[c] = 0
        else: avgprecs[c] = tempAvgprecs

    return avgprecs, np.mean(losses), concat_labels, concat_pred, fnames





def traineval2_model_nocv(dataloader_train, dataloader_test ,  model ,  criterion, optimizer, scheduler, num_epochs, device, numcl):
  best_measure = 0
  best_epoch =-1
  best_measure_array = []

  mAPs=[]
  trainlosses=[]
  testlosses=[]
  testperfs=[]
  zipped = []

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    if(sys.argv[1]!='5'):
        avgloss=train_epoch(model,  dataloader_train,  criterion,  device , optimizer )
        trainlosses.append(avgloss)
        if scheduler is not None:
            scheduler.step()


    perfmeasure, testloss, concat_labels, concat_pred, fnames  = evaluate_meanavgprecision(model, dataloader_test, criterion, device, numcl)
    testlosses.append(testloss)
    testperfs.append(perfmeasure)

    print('at epoch: ', epoch,' classwise perfmeasure ', perfmeasure)

    avgperfmeasure = np.mean(perfmeasure)
    mAPs.append(avgperfmeasure)
    print('at epoch: ', epoch,' avgperfmeasure ', avgperfmeasure)

    if avgperfmeasure > best_measure: #higher is better or lower is better?
      bestweights = model.state_dict()

      #torch.save(model.state_dict(), os.path.join('model.pt'))           #Model is saved here Commented out since i assume
      #TODO track current best performance measure and epoch
      best_measure = avgperfmeasure
      best_epoch = epoch
      best_measure_array = perfmeasure
      print('current best', best_measure, ' at epoch ', best_epoch)

      #TODO save your scores
      #testperfs.append(perfmeasure[best_epoch])
      zipped = zip(fnames, concat_labels, concat_pred)


  textfile = open("yourpredictions.txt", "w")
  for element in list(zipped):
      for item in element:
          test = str(item).replace("\n", "")            # A lot of ugly processing to make the data parsable by tailacc func
          test = test.replace("[", "")
          test = test.replace("]", "")
          test = test.replace("  ", " ")
          if(test[0]=='C'):                             # Dont print true path to file, only filename
              test = test.split("/")
              textfile.write(test[4] + "\n")
          else: textfile.write(test + "\n")
  textfile.close()

  #tailacc(numcl, plot = True)
  reproductionRoutine(numcl, best_measure_array)

  return best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs, mAPs


# Used to fetch the scores from file
def fetchScores(numcl):
    filenames = []
    concat_pred = np.empty((0, numcl))
    concat_labels = np.empty((0, numcl))
    count = 0
    with open('predictions.txt') as f:                                           # Read in valued and place them into a 2d array.
       for line in f:
           if(count == 3):
               count = 0
           count+=1
           if(count==1):
               filenames.append(line)
           else:
               line = line.replace("\n", "")                                     #processing string version of array, guaranteed to be suboptimal, but functionality > performance atm
               nuArray = line.split(" ")
               nuArray = list(filter(str.strip, nuArray))
               nuArray = np.array(nuArray).astype(float)
               if(count==2):
                   concat_labels = np.concatenate((concat_labels, [nuArray]))
               else:
                   concat_pred = np.concatenate((concat_pred, [nuArray]))
    return filenames, concat_pred, concat_labels

# Full implementation of Tailacc functionality for task 2
def tailacc(numcl, plot = True):
    filenames = []
    concat_pred = np.empty((0, numcl))
    concat_labels = np.empty((0, numcl))
    avgprecs = np.zeros(numcl)
    classes, _ = get_classes_list()
    count = 0

    print("Fetching scores from best model")
    with open('predictions.txt') as f:                                           # Read in valued and place them into a 2d array.
       for line in f:
           if(count == 3):
               count = 0
           count+=1
           if(count==1):
               filenames.append(line)
           else:
               line = line.replace("\n", "")                                     #processing string version of array, guaranteed to be suboptimal, but functionality > performance atm
               nuArray = line.split(" ")
               nuArray = list(filter(str.strip, nuArray))
               nuArray = np.array(nuArray).astype(float)
               if(count==2):
                   concat_labels = np.concatenate((concat_labels, [nuArray]))
               else:
                   concat_pred = np.concatenate((concat_pred, [nuArray]))

    for c in range(numcl):
        #avgprecs[c]= "# TODO, nope it is not sklearn.metrics.precision_score"
        val_true, val_pred = concat_labels[:,c], concat_pred[:,c]
        tempAvgprecs = sklearn.metrics.average_precision_score(val_true, val_pred)
        if(np.isnan(tempAvgprecs)): avgprecs[c] = 0
        else: avgprecs[c] = tempAvgprecs

    print("")
    print("Average Precision Scores of Best Epoch (also used in context of reproduction):")
    print(avgprecs)
    print("---------------------------------------")
    topClass = np.where(avgprecs == np.amax(avgprecs))
    print("Highest AP Class = ", classes[topClass[0][0]], ", (index: ", topClass[0][0], ")")

    zipped = zip(filenames, concat_labels, concat_pred)

    ###
    ### Find and show top and bottom 10 images for the highest AP class
    ###
    topten = []
    botten = []
    sortedZip = sorted(zipped, key=lambda x: x[2][topClass[0][0]], reverse=True)                # Sort all images according to specific class AP
    for i in range(10): topten.append(sortedZip[i])                                             # Store 10 highest AP images for specific class
    for i in range(len(sortedZip)-1, len(sortedZip)-11, -1): botten.append(sortedZip[i])        # And the lowest 10



    if(plot):
        fig=plt.figure(figsize=(9,9))
        plt.title(str(classes[topClass[0][0]]), y=1.08)
        plt.axis('off')

        for i in range(1, 21):
            imagepath = []
            if(i<11): imagepath = topten[i-1][0].replace('\n','')
            else: imagepath = botten[i-11][0].replace('\n','')
            nuImgPath = imagepath.replace('tif','jpg')
            img = Image.open(RDIR+'train-jpg/train-jpg/'+nuImgPath)
            subp = fig.add_subplot(4, 5, i)
            subp.axis('off')
            if(i<11): x=topten[i-1][2]
            else:  x=botten[i-11][2]
            subp.title.set_text("Pred: " + str(x[topClass[0][0]]))
            plt.imshow(img)
        plt.show()

    ###
    ### Tailacc section of Task 2
    ###
    tsteps = []
    accuracies = []
    t_max = np.amax(concat_pred[:,c])*100
    t_max = t_max.astype(np.int64)
    for T in range(0,t_max):                              #For every 5th T between 1 and 100
        cAccuracies = []

        for c in range(numcl):                              #For every class
            y_pred = concat_pred[:,c]                       #Copy all predictions for specific class
            y_true = concat_labels[:,c]                     #Copy all labels for specific class
            zipped = zip(y_true, y_pred)
            filteredZip = [(x, y) for x, y in zipped if y > T/100]  #Create a subset with only predictions higher than the threshold
            if(filteredZip == []): continue                 #In case no true positives left
            y_true, y_pred = zip(*filteredZip)
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            tempacc = (y_pred == y_true).sum() / len(y_true)
            cAccuracies.append(tempacc)
        accuracies.append(np.mean(cAccuracies))
    print(np.mean(accuracies))

    bottom = [x/100 for x in range(t_max)]
    if(plot):
        fig=plt.figure(figsize=(9,9))
        plt.title("Tailacc(T), T=0.00..."+str(t_max/100)+" (t_max)", y=1.08)
        plt.plot(bottom, accuracies, label='Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('T')
        plt.show()


def reproductionRoutine(numcl, newScores):
    print("\n\n####################################")
    print("Reproduction Routine Running")
    filenames = []
    concat_pred = np.empty((0, numcl))
    concat_labels = np.empty((0, numcl))
    bestModelScores = np.zeros(numcl)
    classes, _ = get_classes_list()

    filenames, concat_pred, concat_labels = fetchScores(numcl)

    for c in range(numcl):
        #avgprecs[c]= "# TODO, nope it is not sklearn.metrics.precision_score"
        val_true, val_pred = concat_labels[:,c], concat_pred[:,c]
        tempAvgprecs = sklearn.metrics.average_precision_score(val_true, val_pred)
        if(np.isnan(tempAvgprecs)): bestModelScores[c] = 0
        else: bestModelScores[c] = tempAvgprecs

    percentageErr = [abs(x-y)/x*100 for x,y in zip(bestModelScores,newScores)]
    percentageErrArr = np.array(percentageErr)
    absoluteErr = np.subtract(newScores, bestModelScores)
    print("Saved Best APs\n", bestModelScores)
    print("Current APs\n",newScores)
    print("\n\n####################################")
    print("Percentage Error of APs (1-(Current/Best))")
    print(percentageErrArr)
    print("\nAbsolute Error of APs (Current-Best)")
    print(absoluteErr)


class yourloss(nn.modules.loss._Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        super(yourloss, self).__init__()
        self.myloss = nn.BCELoss()
        return
    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        #TODO
        output = self.myloss(input_, target)
        return output



##
##
##      MAIN
##
##
def runstuff():
  config = dict()
  config['use_gpu'] = True #True #TODO change this to True for training on the cluster
  config['lr'] = 0.005
  config['batchsize_train'] = 32
  config['batchsize_val'] = 64
  config['maxnumepochs'] = 2 #35
  config['scheduler_stepsize'] = 10
  config['scheduler_factor'] = 0.3
  config['numcl'] = 17 # This is a dataset property.

  print(sys.argv)

  # Data augmentations.
  data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          #transforms.Normalize([0.7476, 0.6534, 0.4757], [0.1677, 0.1828, 0.2137]),
          transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284]),
      ]),
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
          transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284]),
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
  resmodel = resnet18(pretrained=True)
  resmodelTwo = resnet18(pretrained=True)
  model = resmodel


  if(sys.argv[1]=='1'):
      print("Task1 Running")
      model = SingleNetwork(resmodel)
  elif(sys.argv[1]=='2'):
      print("Task2 (Best predictions from saved model)")
      tailacc(config['numcl'])
  elif(sys.argv[1]=='3'):
      print("Task3 Running")
      model = TwoNetworks(resmodel, resmodelTwo)
  elif(sys.argv[1]=='4'):
      print("Task4 Running")
      model = SingleNetwork(resmodel, "kaiminghe")
  elif(sys.argv[1]=='5'):
      print("Validation Running (Pretrained model)")                                     # Validation of Task 1
      model = SingleNetwork(resmodel)
      model.load_state_dict(torch.load('model.pt'))



  model = model.to(device)
  #for param in model.parameters():
    #print(param)

  #lossfct = nn.BCELoss()
  lossfct = yourloss()
  #TODO
  # Observe that all parameters are being optimized
  someoptimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
  # Decay LR by a factor of 0.3 every X epochs
  #TODO
  #somelr_scheduler = torch.optim.lr_scheduler.LinearLR(someoptimizer, start_factor=config['scheduler_factor'], total_iters=config['maxnumepochs'])
  somelr_scheduler = torch.optim.lr_scheduler.StepLR(someoptimizer, step_size=config['scheduler_stepsize'], gamma=config['scheduler_factor'], verbose=True)
  if(sys.argv[1]!='2'):
      best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs, mAPs = traineval2_model_nocv(dataloaders['train'], dataloaders['val'] ,  model ,  lossfct, someoptimizer, somelr_scheduler, num_epochs= config['maxnumepochs'], device = device , numcl = config['numcl'] )
      fig = plt.figure()
      plt.title("Train & Test Loss Curve")
      plt.plot(trainlosses, label='train')
      plt.plot(testlosses, label='val')
      leg = plt.legend(loc='upper right')
      plt.ylabel('Loss')
      plt.xlabel('Epochs')
      fig.show()
      plt.show()

      fig = plt.figure()
      plt.title("Test mAP Curve")
      plt.plot(mAPs, label='mAP Score')
      leg = plt.legend(loc='upper right')
      plt.ylabel('mAP')
      plt.xlabel('Epochs')
      fig.show()
      plt.show()




if __name__=='__main__':
  np.random.seed(0)
  torch.manual_seed(0)


  runstuff()
