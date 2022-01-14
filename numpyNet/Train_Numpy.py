"""
The training code lifes in this file. 
First I import all the custom written classes and then I try and train the network on the cats vs dogs dataset
"""


import numpy as np
import matplotlib.pyplot as plt
import dill

from imports.layer_conv_V3 import Layer_conv
from imports.layer_dense import Layer_Dense
from imports.maxpool import maxpool
from imports.activation_relu import Activation_ReLU
from imports.activation_softmax_loss_categoricalcrossentropy import (
    Activation_Softmax_Loss_CategoricalCrossentropy,
)
from imports.optimizer_sgd import Optimizer_SGD
from imports.flatten import flatten
from imports.npcustomdataset import CustomImageDataLoader
from imports.activation_softmax import Activation_Softmax

# imports all the custom written classses and matplot+numpy


# testing the dataloader


def labelToClass(label):
    if label == 0:
        return "cat"
    else:
        return "dog"


# plot 6 examples from the data
# fig, ax = plt.subplots(2,3)
# #ax[0,0].imshow(train_features[0].permute(1,2,0))
# #ax[0,0].set_title('dog')
# for row in range(2):
#     for col in range(3):
#         ax[row,col].imshow(train_features[:,:,:,row+3*col])
#         ax[row,col].set_title(labelToClass(train_labels.iloc[row+3*col]))
# plt.show()

##############################################################################################################################################################################
# Now the data is loaded into train_features as a 256,256,3,64 array and the labels are in a pd dataframe
# Time to get to work on the neural network archetecture
# For simplicity I will only create 1 VGG block and some fully connected layers to proofe learning
##############################################################################################################################################################################

conv1 = Layer_conv(32, 3, 3)
relu1 = Activation_ReLU()
max1 = maxpool(4, 4)
flatten1 = flatten()
dense1 = Layer_Dense(131072, 100)
relu2 = Activation_ReLU()
dense2 = Layer_Dense(100, 2)
softmax1 = Activation_Softmax()
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer_mgd = Optimizer_SGD(learning_rate=0.001, momentum=0.9)
#actually this uses minibatch gradient descent since we are doing it over a batch 

#this returns an iterable object that outputs batchsize images and there label until it is exhausted

trainLoader = CustomImageDataLoader(
    "numpyNet/data/dataloader_train.csv", "numpyNet/data/train_renamed", batchsize=64
)


# begining the training loop

monitor = np.zeros((2, 2000))

for epoch in range(1):
    running_loss = 0.0
    for iter, data in enumerate(trainLoader, start=0):
        # you can think of data as next(iter(DataLoader)) because enumerate calls next in a for loop
        inputs, labels = data

        truth = np.zeros((64, 2))
        for n in range(0, len(truth)):
            if labels.iloc[n] == 1:
                truth[n, 0] = 1
            else:
                truth[n, 1] = 1
        # normalizing the inputs by subtracting mean and dividing by sd
        # output[channel] = (input[channel] - mean[channel]) / std[channel]

        inputs = (inputs - np.mean(inputs, axis=(0, 1))) / np.std(inputs, axis=(0, 1))
        # print(np.mean(inputs, axis = (0,1)))
        # print(np.std(inputs, axis = (0,1)))

        # now data goes through the CNN and returns class probabilities
        conv1.forward(inputs)
        relu1.forward(conv1.output)
        max1.forward(relu1.output)
        flatten1.forward(max1.output)
        dense1.forward(flatten1.output)
        relu2.forward(dense1.output)
        dense2.forward(relu2.output)
        loss_activation.forward(dense2.output, truth)
        print("before backprob")
        # now I need to do back prop
        loss_activation.backward(loss_activation.output, truth)
        dense2.backward(loss_activation.dinputs)
        relu2.backward(dense2.dinputs)
        dense1.backward(relu2.dinputs)
        flatten1.backward(dense1.dinputs)
        max1.backward(flatten1.dinputs)
        relu1.backward(max1.dinputs)
        conv1.backward(relu1.dinputs)

        # In the backprob the gradients are stored in dweights and dbiases
        # now the optimizer needs to step and we can repeat
        print("before optim")
        optimizer_mgd.post_update_params()
        optimizer_mgd.update_params(conv1)
        optimizer_mgd.update_params(dense1)
        optimizer_mgd.update_params(dense2)
        optimizer_mgd.post_update_params
        print("after optim")
        # calculate accuracy
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(truth.shape) == 2:
            truth = np.argmax(truth, axis=1)
        # if len(y.shape) == 1:
        #    y = np.argmax(y,axis = 0)
        # only works with 2D y but i use a single one now so axis-1
        accuracy = np.mean(predictions == truth)
        loss = loss_activation.outputloss
        print(f"Current interation is: {iter}")
        print(f"Current accuracy is: {accuracy}")
        print(f"Current loss is: {loss}")
        print(f"Filter new {conv1.weights[0,0,0,0]} dweight {conv1.dweights[0,0,0,0]}")

        monitor[1, iter + 359 * epoch] = accuracy
        monitor[0, iter + 359 * epoch] = loss

fig, ax = plt.subplots(2, 1)
ax[0].plot(np.arange(iter), monitor[0, 0:iter])
ax[0].set_title("loss")
ax[1].plot(np.arange(iter), monitor[1, 0:iter])
ax[1].set_title("Accuracy")
plt.savefig("numpy_training.pdf")
fig.show()


Numpy_net = [conv1, relu1, max1, flatten1, dense1, relu2, dense2, loss_activation]


def save_object(obj, filename):
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        dill.dump(obj, outp)


save_object(Numpy_net, "Numpy_net.pkl")

