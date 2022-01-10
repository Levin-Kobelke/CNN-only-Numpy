# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 12:26:04 2022
Torch skript to train a CNN on cats vs dogs as a control to a custom
build CNN in numpy
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
import os
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
class simpleCNN(nn.Module):
    def __init__(self):
        super(simpleCNN, self).__init__()
        self.LayerStack=nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.Flatten(),
            nn.Linear(238144, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax()
            )
    def forward(self,Input):
        probs=self.LayerStack(Input)
        return probs
    
model = simpleCNN().to('cpu')

#probs is the output after the softmax. So the model probabilities
# for the 2 classes

#now I will define the optimizer and loss criterion

#categorical cross entropy

criterion = nn.CrossEntropyLoss()

#optimizer is SGD

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#mointoring the loss and accuracy per iteration 
monitor = np.zeros((2,2000))

for epoch in range(5):
    running_loss=0.0
    for iter, data in enumerate(train_dataloader,start=0):
        #you can think of data as next(iter(DataLoader)) because enumerate calls next in a for loop
        inputs, labels = data

        optimizer.zero_grad()
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
ax[0].plot(np.arange(360),monitor[0,0:360])
ax[0].set_title('loss')
ax[1].plot(np.arange(360),monitor[1,0:360])
ax[1].set_title('Accuracy')
plt.savefig('foo.png')
plt.savefig('foo.pdf')

fig.show()

fig, ax = plt.subplots(2,1)
ax[0].plot(np.arange(1439),monitor[0,:1439])
ax[0].set_title('loss')
ax[1].plot(np.arange(1439),monitor[1,:1439])
ax[1].set_title('Accuracy')
plt.savefig('foo2.png')
plt.savefig('foo2.pdf')
fig.show()

PATH = './dogs_Torch_net.pth'
torch.save(model.state_dict(), PATH)