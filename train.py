
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import cv2
import copy
from tqdm import tqdm
import csv
import shutil
from PIL import ImageFile


TRAIN = True
TEST = False
CALCULATE_MEAN_AND_STD = False


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

if CALCULATE_MEAN_AND_STD:
    data_transforms = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    data_dir = 'data/'
    #dataset = datasets.ImageFolder(os.path.join(data_dir, "train"),  data_transforms)
    #mean_train, std_train =  get_mean_and_std(dataset)
    #print(mean_train)
    #print(std_train)
    dataset = datasets.ImageFolder(os.path.join(data_dir, "val"),  data_transforms)
    mean_val, std_val =  get_mean_and_std(dataset)
    print(mean_val)
    print(std_val)



# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        #transforms.Resize((330,600)),
        transforms.Resize(224),
        #transforms.CenterCrop(224),
        #transforms.RandomResizedCrop(224),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.Resize((330,600)),
        transforms.Resize(224),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = './'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer,  num_epochs=300):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                pass
                #scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, "checkpoint_epoch_" + str(epoch) + ".pth")


            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft = models.resnet50(pretrained='imagenet')
num_ftrs = model_ft.fc.in_features

#for param in model_ft.parameters():
#  param.require_grad = False
  
fc = nn.Sequential(
    nn.Linear(num_ftrs, 460),
    nn.ReLU(),
    nn.Dropout(0.4),
    
    nn.Linear(460,4),
    nn.LogSoftmax(dim=1)
)
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).


model_ft.fc = fc
model_ft = model_ft.to(device)

criterion = nn.NLLLoss()
#criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.00002, weight_decay=5e-5)
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)


if TRAIN:
    torch.backends.cudnn.enabled = False
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    model_ft = train_model(model_ft, criterion, optimizer_ft,  num_epochs=300)
elif TEST:
    batch_size = 32
    output_list = [["guid/image", "label"]]
    for i, (images, labels) in enumerate(dataloaders["val"], 0):
        images = images.to(device)
        # outputs = model(images)
        # _, predicted = torch.max(outputs.data, 1)
        checkpoint = torch.load("checkpoint_epoch_36.pth")
        model_ft.load_state_dict(checkpoint['model_state_dict'])
        optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
        sample_fname = dataloaders["val"].dataset.samples[batch_size*i:batch_size*i+batch_size]
        outputs = model_ft.forward(images)
        _, preds = torch.max(outputs, 1)

        for i in range(len(sample_fname)):
            filename = sample_fname[i][0]
            filename = filename[filename.rindex('/')+1:]
            filename = filename[:-14] + "/" + filename[-14:-10]
            label = preds[i].item()
            output_list.append([filename, str(label)])

    with open("out.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(output_list)




