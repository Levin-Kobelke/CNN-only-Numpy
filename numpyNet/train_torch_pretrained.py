# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 13:51:22 2022
Torch skript to train a CNN on cats vs dogs as a control to a custom
build CNN in numpy. This CNN is more complex than current np version. It uses an pretrained model and only resets the FC

works really well
@author: Levin_user
"""
# =============================================================================
# To Do
#1) Define Dataloaders to load data
#2) Build Architecture with torch.nn modules
#3) define traning loop
#4) optional: Hyperparameter optimisation
# =============================================================================

#dataloaders
import torch
import matplotlib.pyplot as plt
from imports.customdataset import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda
import torch.optim as optim
import numpy as np
from torchvision import models
# =============================================================================
# ToDo

#1) implement normalization of images for pretrained model
#2) loadpretrained model and reset last layer

# =============================================================================
#Custom image Dataset is a class containing the necessary functions
#to make a dataset usable for the pytorch dataloader




#I use transforms to bring image data into torch tensor format
#I use target_transform to bring integer classes into one hot vector
training_data = CustomImageDataset(
    './numpyNet/data/dataloader_train.csv',
    './numpyNet/data/train_renamed/',
    transform=Lambda(lambda x: x.type(torch.float32)),
    target_transform=Lambda(lambda y: torch.zeros(2,
                                                  dtype=torch.float)
                            .scatter_(dim=0, index=y,value=torch.tensor(1, dtype=torch.uint8)))
    )
                                   
test_data = CustomImageDataset(
    './numpyNet/data/dataloader_test.csv',
    './numpyNet/data/test_renamed/',
    #transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(2,
                                                  dtype=torch.float)
                            .scatter_(dim=0, index=y,value=1))
    )  
                               

#using the torch dataloader class to define a object for training
#and testing data. This allows for iterating through the data
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

#testing the dataloader

train_features, train_labels = next(iter(train_dataloader))
#defining the class
def labelToClass(label):
    if(label ==0):
        return 'cat'
    else:
        return 'dog'
#plot 6 examples from the data
# fig, ax = plt.subplots(2,3)
# #ax[0,0].imshow(train_features[0].permute(1,2,0))
# #ax[0,0].set_title('dog')
# for row in range(2):
#     for col in range(3):
#         ax[row,col].imshow(train_features[row+3*col].permute(1,2,0))
#         ax[row,col].set_title(labelToClass(train_labels[row+3*col]))
# plt.show()



# =============================================================================
# Now I want to define a class including all the modules
# I need for my CNN. Thus, I define the Architecture here
# =============================================================================
from torch import nn
model = models.resnet18(pretrained=True)
in_features_fc = model.fc.in_features
model.fc = nn.Linear(in_features_fc, 2)

model = model.to('cpu')

probs = model(train_features)
#probs is the output after the softmax. So the model probabilities
# for the 2 classes

#now I will define the optimizer and loss criterion

#categorical cross entropy

criterion = nn.CrossEntropyLoss()

#optimizer is SGD

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#mointoring the loss and accuracy per iteration 
monitor = np.zeros((2,2000))

for epoch in range(1):
    running_loss=0.0
    for iter, data in enumerate(train_dataloader,start=0):
        #you can think of data as next(iter(DataLoader)) because enumerate calls next in a for loop
        inputs, labels = data
        #now data goes through the CNN and returns class probabilities
        probs=model(inputs)
        #categorical cross entropy to calculate the loss
        loss = criterion(probs, labels)
        #now I need to do back prop. this is insanly easy using the torch framework
        loss.backward()
        #finally the gradient needs to step once per iteration since we do SGD
        #insane how all parameters are updated by one call to the optimizer. frameworks are insane
        optimizer.step()
        
        #basically this is enough but I want to record some statistics
        
        
        correct=0
        total=0
        _,predicted = torch.max(probs.data,1)
        total=labels.size(0)
        correct=(labels[:,1]==predicted).sum()
        accuracy=correct/total
        monitor[1,iter+360*epoch]=accuracy
        monitor[0,iter+360*epoch]=loss
        print(f'loss {loss} accuracy {accuracy} iteration {iter}')

fig, ax = plt.subplots(2,1)
ax[0].plot(np.arange(iter),monitor[0,0:iter])
ax[0].set_title('loss')
ax[1].plot(np.arange(iter),monitor[1,0:iter])
ax[1].set_title('Accuracy')
plt.savefig('pretrained.png')
plt.savefig('pretrained.pdf')
fig.show()

fig, ax = plt.subplots(2,1)
ax[0].plot(np.arange(1439),monitor[0,:1439])
ax[0].set_title('loss')
ax[1].plot(np.arange(1439),monitor[1,:1439])
ax[1].set_title('Accuracy')
fig.show()

PATH = './dogs_Torch_net_res.pth'
torch.save(model.state_dict(), PATH)
