"""
The training code lifes in this file. 
First I import all the custom written classes and then I try and train the network on the cats vs dogs dataset
"""


import numpy as np
import matplotlib.pyplot as plt

from imports.Layer_conv import Layer_conv
from imports.Layer_dense import Layer_Dense
from imports.maxpool import maxpool
from imports.Activation_ReLU import Activation_ReLU
from imports.Activation_Softmax_Loss_CategoricalCrossentropy import Activation_Softmax_Loss_CategoricalCrossentropy
from imports.Optimizer_SGD import Optimizer_SGD
from imports.flatten import flatten
from imports.npcustomdataset import CustomImageDataLoader
from imports.Activation_Softmax import Activation_Softmax

#imports all the custom written classses and matplot+numpy




#testing the dataloader


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
#         ax[row,col].imshow(train_features[:,:,:,row+3*col])
#         ax[row,col].set_title(labelToClass(train_labels.iloc[row+3*col]))
# plt.show()

##############################################################################################################################################################################
#Now the data is loaded into train_features as a 256,256,3,64 array and the labels are in a pd dataframe
#Time to get to work on the neural network archetecture
#For simplicity I will only create 1 VGG block and some fully connected layers to proofe learning
##############################################################################################################################################################################

Conv1 = Layer_conv(32, 3, 3)
ReLU1 = Activation_ReLU()
Max1 = maxpool(4,4)
Flatten1 = flatten()
Dense1 = Layer_Dense(131072,100)
ReLU2 = Activation_ReLU()
Dense2 = Layer_Dense(100,2)
Softmax1 = Activation_Softmax()
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
Optimizer_SGD = Optimizer_SGD(learning_rate=0.001, momentum=0.9)

#this returns an iterable object that outputs batchsize images and there label until it is exhausted
TrainLoader = CustomImageDataLoader('numpyNet/data/dataloader_train.csv','numpyNet/data/train_renamed', batchsize=64)



#begining the training loop

monitor = np.zeros((2,2000))

for epoch in range(1):
    running_loss=0.0
    for iter, data in enumerate(TrainLoader,start=0):
        #you can think of data as next(iter(DataLoader)) because enumerate calls next in a for loop
        inputs, labels = data

        truth = np.zeros((64,2))
        for n in range(0,len(truth)):
            if labels.iloc[n] == 1:
                truth[n,0] = 1
            else:
                truth[n,1] = 1

        #now data goes through the CNN and returns class probabilities
        Conv1.forward(inputs)
        ReLU1.forward(Conv1.output)
        Max1.forward(ReLU1.output)
        Flatten1.forward(Max1.output)
        Dense1.forward(Flatten1.output)
        ReLU2.forward(Dense1.output)
        Dense2.forward(ReLU2.output)
        loss_activation.forward(Dense2.output, truth)

        #now I need to do back prop
        loss_activation.backward(loss_activation.output, truth)
        Dense2.backward(loss_activation.dinputs)
        ReLU2.backward(Dense2.dinputs)
        Dense1.backward(ReLU2.dinputs)
        Flatten1.backward(Dense1.dinputs)
        Max1.backward(Flatten1.dinputs)
        ReLU1.backward(Max1.dinputs)
        Conv1.backward(ReLU1.dinputs)

        #In the backprob the gradients are stored in dweights and dbiases
        #now the optimizer needs to step and we can repeat
        Optimizer_SGD.post_update_params()
        Optimizer_SGD.update_params(Conv1)
        Optimizer_SGD.update_params(Dense1)
        Optimizer_SGD.update_params(Dense2)
        Optimizer_SGD.post_update_params
                
        #calculate accuracy
        predictions = np.argmax(loss_activation.output, axis = 1)
        if len(truth.shape) == 2:
            truth = np.argmax(truth,axis = 1)
        #if len(y.shape) == 1:
        #    y = np.argmax(y,axis = 0)  
        #only works with 2D y but i use a single one now so axis-1
        accuracy = np.mean(predictions == truth)
        loss = loss_activation.outputloss
        print(f'Current interation is: {iter}')
        print(f'Current accuracy is: {accuracy}')
        print(f'Current loss is: {loss}')
        print(f'Filter new {Conv1.weights[0,0,0,0]} dweight {Conv1.dweights[0,0,0,0]}')
 

        monitor[1,iter+359*epoch]=accuracy
        monitor[0,iter+359*epoch]=loss

fig, ax = plt.subplots(2,1)
ax[0].plot(np.arange(iter),monitor[0,0:iter])
ax[0].set_title('loss')
ax[1].plot(np.arange(iter),monitor[1,0:iter])
ax[1].set_title('Accuracy')
fig.show()
