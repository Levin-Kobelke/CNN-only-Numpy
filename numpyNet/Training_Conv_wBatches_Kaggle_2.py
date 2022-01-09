# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:17:35 2021
@author: Levin user

Training a simple NN with convoluted images
Second batch of images

"""

def WithOptimizationLearningRateDecayMomentum(minibatchsize, Nsteps):
    #difference here is that I added backprop (hence derivatives/gradients) to
    #each method
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from skimage import io
    import dill
    #for testing
    #import nnfs #a data set we call from a python package. don't forget 
                #to run "pips install nnfs" first
    #from nnfs.datasets import spiral_data
    
    
    train_df = pd.DataFrame({"file": os.listdir("C:/Users/Levin_user/Desktop/Kaggle/train/train")})
    train_df["label"] = train_df["file"].apply(lambda x: x.split(".")[0])    
    train_df["binary"] = np.zeros((25000,1))   
    for img in range(0,len(train_df)):
        if train_df['label'][img] == 'dog':
            train_df['binary'][img]=1

    test = train_df.sample(25000)


    truth = np.zeros((25000,2))
    #we want a array holding [1,0] for good and [0,1] for bad
    for n in range(0,len(truth)):
        if test['binary'].iloc[n] == 1:
            truth[n,0] = 1
        else:
            truth[n,1] = 1
    #df_all['good_ar'] = np.ndarray.tolist(truth)
    train_df['truth']=np.ndarray.tolist(truth)
    #now shuffel the data
    
    
    #now we want a for loop getting the data and running it
      #creating the actual data:
    #x: 3x200 data points (2D)
    #y: corresponding classes
    #[x, y] = spiral_data(samples = 200, classes = 3)
    #file_name = "C:\\Users\\Levin_user\\Desktop\\CNN_splits\\"+"batch_" + str(0) +'.npy'
    
    
    
    
    #for momentum in list([[0.01],[0.02],[0.04],[0.08],[0.16],[0.32],[0.64],[1.28]]):
    #for neurons in list([50,100,150,200,250,300]):
    for neurons in list([100]):
    
    #if you use for loop here you need to indent!
        
        
        #nnfs.init()
        
                
        class Layer_conv:
            """
            Should take as input data from the previous layer and return one value
            All weights of one filter learned together
            define stepsize and padding
            define filter size
            filters are randomly innitiated
            """
                   
                
            def __init__(self, n_neurons = 1,input_channels = 3,size = 3):
                """
                Initalizes convolution layer
        
                Parameters
                ----------
                n_neurons : int, optional
                    How many different classes of neurons to create. The default is 0, because you are supposed to use first in the
                    case you don't want to set a specific amounts of neuron classes.
                input_channels: int
                    Number of channels the input has. Channels output = n_neurons!
                size : int, odd, optional
                    kernel size for the convolution. The default is 3.
                Returns
                -------
                None.
        
                """
                filter_D19_file = np.load('darknet19.weights.npz')
                D19_0conv = filter_D19_file.get('0-convolutional/kernel:0')

        
                
              
                self.filter_size = size
                self.n_neurons = n_neurons
                self.input_channels = input_channels
                kernels = np.random.default_rng().uniform(-0.9,0.9,size=(size,size,input_channels,n_neurons))#init random kernel
                biases = np.random.default_rng().uniform(-0.9,0.9,size=(input_channels,n_neurons))
                
                
                #self.weights = kernels * D19_0conv[:,:,:,0:n_neurons]
                self.weights =kernels             
                #self.weights = D19_0conv[:,:,:,0:n_neurons]
                #lets try starting with pretrained weights
                self.biases = biases
                    #better would be xaviar initalizatio or similar
   
            def forward(self, inputs):
                """
                keyword arguments
                inputs: input image data batch in 4. Dimension of array
                self: self object with filters and bias to apply
                function calculates covolution with input for each filter
                """
                from conv_with_padding_3 import conv_with_padding_3
                #imports conv function
                
                self.output = np.zeros((inputs.shape[0],inputs.shape[1],self.n_neurons,inputs.shape[3]))
                #initalizes results
                #initalizes intermediated array
                #in retrospective filter number coresponds to the activ neuron so it should be renamed
                self.inputs = inputs
                for image in range(0,np.shape(inputs)[3]):
                    #loops over each input 
                    for filter_num in range(0,self.n_neurons):
                        conv_map = np.zeros((inputs.shape[0],inputs.shape[1],self.input_channels))
                        for channel in range(0,self.input_channels):            
                            curr_filter = self.weights[:,:,:,filter_num] # getting a filter from the bank.  
                            #doing the actual convolution layer wise
                            conv_map[:,:,channel] =conv_with_padding_3(inputs[:,:,channel,image], curr_filter[:,:,channel])
                            conv_map[:,:,channel] +=self.biases[channel,filter_num]
                            
                        conv_map = np.average(conv_map,axis=2)
                        self.output[:,:,filter_num,image]=conv_map # summing map with the current filter.
                        
        
            # defining our activation function      
            def backward(self, dvalues):
                #I want to multiplz the dvalues for a filter with all input channels
                from conv_with_padding_3 import conv_with_padding_3
        
                
                #then I sum over the multiplicated image dependend on which weight i 
                #want to find. the x,z directions shift the sum in one channel while the
                #filter channel corresponds to the channel of the image
                
                #write into a matrix and done
                nr,nc,nd,nI = np.shape(self.inputs)
                # npad is a tuple of (n_before, n_after) for each dimension
                npad = ((int(self.filter_size/2), int(self.filter_size/2)), (int(self.filter_size/2), int(self.filter_size/2)), (0, 0),(0,0))
                inp_padded =np.pad(self.inputs,pad_width=npad,mode = 'constant',constant_values =0)
                dweights = np.empty((self.filter_size,self.filter_size,self.input_channels,self.n_neurons,self.inputs.shape[3]))
                for image in range(0,self.inputs.shape[3]):
                    for neuron in range(0,self.n_neurons):
                        for channel in range(0,self.input_channels):
                            for i in range(0,self.filter_size):
                                for j in range(0,self.filter_size):
                                    intermediate = 1/self.n_neurons * dvalues[:,:,neuron,image] * inp_padded[i:i+nr,j:j+nc,channel,image]
                                    #intermediate is an array that holds the product of every input 
                                    #channel with the output. So all inputs get multiplied to the same
                                    #dvalue since they are combined in the same output
                                    #to accuretly multiplie the correct gradient with the correct inputs
                                    #the inp_padded selection is shifted with each intermediate and the
                                    #multiplication window is always as big as the dvalues for 1 neuron
                                    

                                    #dweights[i,j,channel,neuron,image] = np.sum(intermediate)
                                    dweights[i,j,channel,neuron,image] = np.mean(intermediate)#i am sure it should be sum\
                                        #but lets try mean
                                    #while each weight contributed to each output they must be multiplied
                                    #with the correct inputs. #what i am doing here might not be
                                    #exact since I am not taking boundary effects into account
                self.dweights = np.average(dweights, axis = 4)
                #########################################
                #old code wrong
                #local_grad = np.zeros((self.filter_size,self.filter_size))
                #channel depth fehlt noch
                #necessary for derivative
                #dweights = np.zeros((self.filter_size,self.filter_size,self.n_neurons))
        
                #pad = int(self.filter_size/2)
                
                #img_pad = np.pad(self.inputs,pad,"symmetric")
                #r,c,d=np.shape(img_pad)
                #since the rate change in z dependent on the weight is only depndent
                #on x the local gradient for all channels is the same
                #a change is only introduced when multiplying with dvalues
                #but x is different in all channels so...
                #for i in range(0,self.filter_size):
                #    for j in range(0,self.filter_size):
                #        local_grad[i,j] = np.sum(img_pad[i:r-2*pad,j:c-2*pad])
                #now we need to multiply the local gradient with the global gradient
                #we multiplie all gradients with one dz/dw from local grad and the sum
                #for channel in range(0,self.n_neurons):
                #    for i in range(0,self.filter_size):
                #        for j in range(0,self.filter_size):
                #            dweights[i,j,channel] = np.sum(dvalues[:,:,channel]*local_grad[i,j])
                
                #self.dweights = dweights
                
                #############################################################
                #Finding the gradient in respect to the bias
                #local gradient is 1 since it is the derivative of a sum to exponent 1
                #local_grad_bias = 1
                #now we multiply the local gradient with the global gradient from upstream
                
                #grad_bias = local_grad_bias * dvalues
                #to save computation this is commented out
                
                #finally we need to sum up the gradients for one neuron
                
                #gradients_bias = np.sum(dvalues, axis = (0,1))
                gradients_bias = np.mean(dvalues, axis = (0,1))

                gradients_bias=np.average(gradients_bias,axis=1)
                #calculate average over images
                #self.dbiases = np.tile(gradients_bias,(3,1))
                self.dbiases = np.tile(gradients_bias,(self.input_channels,1))

                
                #finally we need the gradient in respect to the inputs
                #should be simmilar to before
                #since every input is multiplied with each weight once (appart from edges)
                #we need to multiplie dvalues at the posstion on which the input had an effect
                #this should be 9 positions for a 3x3 filter and then sum over them
                #also we should divide the gradient by nine because of the summation.
                #Interestingly this is a convolution with a flipped filter
                #so we need to calculate the convolution with each channel of a flipped filter
                #and each filter. Then sum up over the different filters in one channel
                self.dinputs = np.zeros_like(self.inputs)
                n,m,k,I = np.shape(self.inputs)
                intermediate_inputs = np.zeros((n,m,k,self.n_neurons,I))
                for image in range(0,I):
                    for neuron in range(0,self.n_neurons):
                        for channel in range(0,self.input_channels):
                            curr_filter = self.weights[:,:,channel,neuron]
                            curr_filter_rotated = np.rot90(curr_filter, 2)
                            intermediate_inputs[:,:,channel,neuron,image]= conv_with_padding_3(dvalues[:,:,neuron,image], curr_filter_rotated)
                    
                #self.dinputs = np.average(np.sum(intermediate_inputs, axis=3),axis=3)
                self.dinputs = np.mean(intermediate_inputs, axis=3)

                # self.dinputs  = np.dot(dvalues, self.weights.T)
            
            
        #definint the flattening method to input convolution into dense layers
        
        class flatten:
            def forward(self, inputs):
                self.output = inputs.reshape(-1,inputs.shape[-1]).T#flattenes array into 1D in cols for dense
                
                self.inputs = inputs
            def backward(self, dvalues):
                nr,nc,nd,nI = np.shape(self.inputs)
                dvalues = dvalues.T
                self.dinputs = dvalues.reshape((nr,nc,nd,nI))
                
        class maxpool:
            def __init__(self,size,stride):
                self.size=size
                self.stride=stride
            def forward(self,inputs):
                from im2col_indices import im2col_indices
                #to copy stanford code in need to reorganize axis
                #input is (h,w,d,n) stanford(n,d,h,w)
                self.inputs=inputs
                size=self.size
                stride=self.stride
                
                X= inputs.transpose(2,3,1,0)
                self.X = X
                n,d,h,w = np.shape(X)
                # First, reshape it to 50x1x28x28 to make im2col arranges it fully in column
                X_reshaped = X.reshape(n * d, 1, h, w)
                self.X_reshaped=X_reshaped
                # The result will be 4x9800
                # Note if we apply im2col to our 5x10x28x28 input, the result won't be as nice: 40x980
                self.X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)
                
                # Next, at each possible patch location, i.e. at each column, we're taking the max index
                self.max_idx = np.argmax(self.X_col, axis=0)
                
                # Finally, we get all the max value at each column
                # The result will be 1x9800
                out = self.X_col[self.max_idx, range(self.max_idx.size)]
                h_out=int(h/size)
                w_out=int(w/size)
                # Reshape to the output size: 14x14x5x10
                out = out.reshape(h_out, w_out, n, d)
                
                # Transpose to get 14,14,10,5 output
                self.output = out.transpose(0, 1, 2, 3)
                #output is (h_out,w_out,d,n) stanford(n,d,h,w)
            def backward(self,dvalues):
                # X_col and max_idx are the intermediate variables from the forward propagation step
                from col2im_indices import col2im_indices
                # Suppose our output from forward propagation step is 5x10x14x14
                # We want to upscale that back to 5x10x28x28, as in the forward step
                
                # 4x9800, as in the forward step
                dX_col = np.zeros_like(self.X_col)
                
                # 5x10x14x14 => 14x14x5x10, then flattened to 1x9800
                # Transpose step is necessary to get the correct arrangement
                dout_flat = dvalues.transpose(0, 1, 3, 2).ravel()
                
                # Fill the maximum index of each column with the gradient
                
                # Essentially putting each of the 9800 grads
                # to one of the 4 row in 9800 locations, one at each column
                dX_col[self.max_idx, range(self.max_idx.size)] = dout_flat
                
                # We now have the stretched matrix of 4x9800, then undo it with col2im operation
                # dX would be 50x1x28x28
                h,w,d,n=np.shape(self.inputs)
                dX = col2im_indices(dX_col, (n * d, 1, h, w), self.size, self.size, padding=0, stride=self.stride)
                
                # Reshape back to match the input dimension: 5x10x28x28
                dX = dX.reshape(self.X.shape)
                self.dinputs = dX.transpose(2,3,0,1)
                
        #we now defines the layers from the previous chapter as classes and can 
        #therefore define weights and biases as method
        #the fully connected (fc) layer we had before is also called "dense layer"
        class Layer_Dense:
        #dense layer is build for more inputs than 1 img
            def __init__(self, n_inputs, n_neurons):
                #note: we are using randn here in order to see if neg values are 
                #clipped by the ReLU
                self.weights = np.random.randn(n_inputs, n_neurons)
                self.biases  = np.zeros((1, n_neurons))
                
        #passing on the dot product as input for the next layer, as before
            def forward(self, inputs):
                self.output  = np.dot(inputs, self.weights) + self.biases
                self.inputs  = inputs#we're gonna need for backprop
                
            def backward(self, dvalues):
                #gradients
                self.dweights = np.dot(np.mat(self.inputs).T,dvalues)
                self.dbiases  = np.sum(dvalues, axis = 0, keepdims = True)
                self.dinputs  = np.dot(dvalues, self.weights.T)
             
        #defining our activation function        
        class Activation_ReLU:
            
            def forward(self, inputs):
                self.output  = np.maximum(0,inputs)
                self.inputs  = inputs
                
            def backward(self, dvalues):
                self.dinputs = dvalues.copy()
                self.dinputs[self.inputs <= 0] = 0#ReLU derivative
                
        class Activation_sigmoid:
            
            def forward(self, inputs):
                self.output  = 1/(1 + np.exp(-inputs-np.max(inputs,axis =1,\
                                                            keepdims = True)))
                #max in order to prevent overflow
                
                #sigmoid funtion
                self.inputs  = inputs
                
            def backward(self,dvalues):
                self.dinputs = self.output * (1-self.output)
                #derivative of a sigmoid
            
        class Activation_Softmax:
      
            def forward(self,inputs):
                self.inputs = inputs
                exp_values  = np.exp(inputs - np.max(inputs, axis = 1,\
                                              keepdims = True))#max in order to 
                                                               #prevent overflow
                #normalizing probs
                probabilities = exp_values/np.sum(exp_values, axis = 1,\
                                              keepdims = True)  
                self.output   = probabilities                                                
            
            def backward(self, dvalues):
                #just initializing a matrix
                self.dinputs = np.empty_like(dvalues)
                
                for i, (single_output, single_dvalues) in \
                    enumerate(zip(self.output, dvalues)):
                    
                    single_output   = single_output.reshape(-1,1)
                    jacobMatr       = np.diagflat(single_output) - \
                                      np.dot(single_output, single_output.T)
                    self.dinputs[i] = np.dot(jacobMatr, single_dvalues)
            
            
         #defining a common loss function that just calls the part loss function
         #and calculates the mean over all samples
        class Loss:
             
             def calculate(self, output, y):
                 
                 sample_losses = self.forward(output, y)
                 data_loss     = np.mean(sample_losses)
                 return(data_loss)
            
            
        class Loss_CategoricalCrossEntropy(Loss): 
                               #y_pred is not the predicted y, it is its 
                               #probability!!
             def forward(self, y_pred, y_true):
                 samples = len(y_pred)
                 #removing vals close to zero and one bco log and accuracy
                 y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
                 
                 #now, depending on how classes are coded, we need to get the probs
                 if len(y_true.shape) == 1:#classes are encoded as [[1],[2],[2],[4]]
                     correct_confidences = y_pred_clipped[range(samples), y_true]
                 elif len(y_true.shape) == 2:#classes are encoded as
                                            #[[1,0,0], [0,1,0], [0,1,0]]
                     correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1)
                 #above code only works if the classes are delivered in a vector
                 #since right now I am testing one by one shape is confused
                     
                 #if len(y_true.shape) == 0:#classes are encoded as [[1],[2],[2],[4]]
                 #    correct_confidences = y_pred_clipped[range(samples), y_true]
                 #elif len(y_true.shape) == 1:#classes are encoded as
                 #                           #[[1,0,0], [0,1,0], [0,1,0]]
                 #    correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1)
                 #print(f'y_pred: {y_pred}\n'+
                 #      f'y_true: {y_true}\n'+
                 #      f'corr_confidences: {correct_confidences}')        
                     
                 #now: calculating actual losses
                 negative_log_likelihoods = -np.log(correct_confidences)
                 return(negative_log_likelihoods)
                 
             def backward(self, dvalues, y_true):
                 Nsamples = len(dvalues)
                 Nlabels  = len(dvalues[0])
                 #turning labels into one-hot i. e. [[1,0,0], [0,1,0], [0,1,0]], if
                 #they are not
                 if len(y_true.shape) == 1:
                    #"eye" turns it into a diag matrix, then indexing via the label
                    #itself
                    y_true = np.eye(Nlabels)[y_true]
                 #normalized gradient
                 self.dinputs = -y_true/dvalues/Nsamples
                 #this is the local derivative
                 
        class Activation_Softmax_Loss_CategoricalCrossentropy():
            
            def __init__(self):
                self.activation = Activation_Softmax()
                self.loss       = Loss_CategoricalCrossEntropy()
                
            def forward(self, inputs, y_true):
                self.activation.forward(inputs)
                self.output = self.activation.output#the probabilities
                #calculates and returns mean loss
                return(self.loss.calculate(self.output, y_true))
                
            def backward(self, dvalues, y_true):
                Nsamples = len(dvalues)
                if len(y_true.shape) == 2:
                    y_true = np.argmax(y_true, axis = 1)
                #if len(y_true.shape) == 1:
                #    y_true = np.argmax(y_true, axis = 0)
                    #y_true assumes an array for for axis = 1 but one bz one axis = old_axis-1
                self.dinputs = dvalues.copy()
                #calculating normalized gradient
                self.dinputs[range(Nsamples), y_true] -= 1
                self.dinputs = self.dinputs/Nsamples
                
                
        class Optimizer_SGD:
            #initializing with a default learning rate of 0.1
            def __init__(self, learning_rate = 0.01, decay = 0, momentum = 0):
                self.learning_rate         = learning_rate
                self.current_learning_rate = learning_rate
                self.decay                 = decay
                self.iterations            = 0
                self.momentum              = momentum
                
            def pre_update_params(self):
                if self.decay:
                    self.current_learning_rate = self.learning_rate * \
                        (1/ (1 + self.decay*self.iterations))
                
            def update_params(self, layer):
                
                #if we use momentum
                if self.momentum:
                    
                    #check if layer has attribute "momentum"
                    if not hasattr(layer, 'weight_momentums'):
                        layer.weight_momentums = np.zeros_like(layer.weights)
                        layer.bias_momentums   = np.zeros_like(layer.biases)
                        
                    #now the momentum parts
                    weight_updates = self.momentum * layer.weight_momentums - \
                        self.current_learning_rate * layer.dweights
                    layer.weight_momentums = weight_updates
                    
                    bias_updates = self.momentum * layer.bias_momentums - \
                        self.current_learning_rate * layer.dbiases
                    layer.bias_momentums = bias_updates
                    
                else:
                    
                    weight_updates = -self.current_learning_rate * layer.dweights
                    bias_updates   = -self.current_learning_rate * layer.dbiases
                
                layer.weights += weight_updates
                layer.biases  += bias_updates
                
                
                
                
            def post_update_params(self):
                self.iterations += 1
                
                
                
                
    ###############################################################################
    
    ###############################################################################
    #1st Layer            
        #we have 2D data (inputs) and three classes (neurons)        
        #dense1          = Layer_Dense(1048576, neurons)
        Conv1           = Layer_conv(n_neurons = 8, input_channels=3, size = 3)
        activation1     =Activation_ReLU()
        maxpool1        =maxpool(stride=2,size=2)
        Conv2           =Layer_conv(n_neurons=16,input_channels=8, size = 3)
        activation2     =Activation_ReLU()
        Conv3           =Layer_conv(n_neurons=16, input_channels=16, size=3)
        activation3     =Activation_ReLU()
        
        Conv4           =Layer_conv(n_neurons=8, input_channels=16, size=3)
        activation4     =Activation_ReLU()

        maxpool2        =maxpool(stride=2,size=2)
        flatten1        = flatten()
        dense1          = Layer_Dense(32768, neurons)
        activation5     = Activation_ReLU()
        #activation1     =Activation_sigmoid()
        optimizer       = Optimizer_SGD(learning_rate= 0.0002, decay = 0.001, momentum = 0.1)
        dense2          = Layer_Dense(len(dense1.biases.T), 2)#two classes  
        loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    #calling loss function
        #loss_function    = Loss_CategoricalCrossEntropy()
        #Nsteps  = 10
        MaxBatches = 100
        Monitor = np.zeros((Nsteps,3))
        #SelMonitor =np.zeros((MaxBatches,2))
        SelMonitor =np.zeros((Nsteps,2))
        
        Epochs = 3
        #fuer eine minibatch
        #I want to try and fit it to one image of each class. See if that works

        #minibatchsize = 40
        #minibatchsize = 10
        #for epoch in range(0,Epochs):#indent here for epochs
        #for batch in range(0,int(25000/minibatchsize)):
            
        for batch in range(0,MaxBatches): #data too massiv to run it all with slow script(?)
   
            for iteration in range(Nsteps):
                #passing data through layer
                x=[]
                for image in range(0,minibatchsize):
                    img =[io.imread('C:/Users/Levin_user/Desktop/Kaggle/train/resized/'+test['file'].iloc[image+batch*minibatchsize])]
                    x.append(img)
                x=np.squeeze(np.stack(x,axis=4))
                y = truth[batch*minibatchsize:(1+batch)*minibatchsize,:]
                
                
               #writing the data into a new array for the dense layer
                Conv1.forward(x)
                activation1.forward(Conv1.output)
                maxpool1.forward(activation1.output)
                Conv2.forward(maxpool1.output)
                activation2.forward(Conv2.output)
                Conv3.forward(activation2.output)
                activation3.forward(Conv3.output)
                Conv4.forward(activation3.output)
                activation4.forward(Conv4.output)
                maxpool2.forward(activation4.output)
                flatten1.forward(maxpool2.output)            
                dense1.forward(flatten1.output)
                activation5.forward(dense1.output)
                dense2.forward(activation5.output)
                loss = loss_activation.forward(dense2.output, y)
               
                predictions = np.argmax(loss_activation.output, axis = 1)
                if len(y.shape) == 2:
                    y = np.argmax(y,axis = 1)
                #if len(y.shape) == 1:
                #    y = np.argmax(y,axis = 0)  
                #only works with 2D y but i use a single one now so axis-1
                accuracy = np.mean(predictions == y)
            
            #backward passes
                loss_activation.backward(loss_activation.output, y)
                dense2.backward(loss_activation.dinputs)
                activation5.backward(dense2.dinputs)
                dense1.backward(activation5.dinputs)
                flatten1.backward(dense1.dinputs)
                maxpool2.backward(flatten1.dinputs)
                activation4.backward(maxpool2.dinputs)
                Conv4.backward(activation4.dinputs)
                activation3.backward(Conv4.dinputs)
                Conv3.backward(activation3.dinputs)
                activation2.backward(Conv3.dinputs)
                Conv2.backward(activation2.dinputs)
                maxpool1.backward(Conv2.dinputs)
                activation1.backward(maxpool1.dinputs)
                Conv1.backward(activation1.dinputs)
                

                
                
                
                
                
                
                optimizer.pre_update_params()#decaying learning rate
                optimizer.update_params(dense1)
                optimizer.update_params(dense2)
                optimizer.update_params(Conv1)#lets see if the conv1 is a problem
                optimizer.update_params(Conv2)
                optimizer.update_params(Conv3)
                optimizer.update_params(Conv4)  
                optimizer.post_update_params()#just increasing iteration by one
            
                Monitor[iteration,0] = accuracy
                Monitor[iteration,1] = loss
                Monitor[iteration,2] = optimizer.current_learning_rate
                
                #if(iteration==0):
                #print('Nsteps=0')
                #SelMonitor[batch,0]=accuracy
                #SelMonitor[batch,1]=loss
                SelMonitor[iteration,0]=accuracy
                SelMonitor[iteration,1]=loss
                #plt.close('all')
                fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True)
                #ax[0].plot(np.arange(batch),SelMonitor[0:batch,0])
                ax[0].plot(SelMonitor[0:iteration,0])
                ax[0].set_ylabel('accuracy [%]')
                #ax[1].plot(np.arange(batch),SelMonitor[0:batch,1])
                ax[1].plot(SelMonitor[0:iteration,1])
                ax[1].set_ylabel('loss')
                plt.title(f'batch={batch}')
                if(0==batch%100):
                    plt.show()    
                    
            #    print(f'Conv1.dweights Img1: {Conv1.dweights[0:10,0,0,0]}\n'+
            #          f'dense1.dinputs: {dense1.dinputs[1,1:5]}\n'+
            #          #f'activation1.dinputs: {activation1.dinputs[1,1:5,0,0]}\n'+
            #          f'dense2.dinputs: {dense2.dinputs[1,1:5]}\n'+
                      #f'loss_activation.dinputs: {loss_activation.dinputs[1,1:5]}\n'+
            #          f'loss_activation.outputs:{loss_activation.output[0:10,0:10]}\n'+
            #          f'truth:      {y[0:10]}\n'+
             #         f'prediction: {predictions[0:10]}'
                      #f'Conv1.dvalues Img2: {Conv1.dweights[0:10,0,0,1]}\n'+
                      #f'flatten.forward: {flatten1.output[0,0:10]}\n'+
                      #f'dense1.forward: {dense1.output[0,0:10]}\n'+
                      #f'activation1.forward: {activation1.output[0,0:10]}\n'+
                      #f'dense2.forward: {dense2.output[0:5,:]}\n'
              #        )
    
                if not iteration % 1:
                    print(f'-------------------------------------------------------\n' +
                          f' batch, iteration: {batch}, {iteration} \n' +
                          f'accuracy: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f}, ' +
                          #f'actual learning rate: {optimizer.current_learning_rate}\n'+
                          #f'Conv1.dweights Img1: {Conv1.dweights[0:10,0,0,0]}\n'+
                          #f'dense1.dinputs: {dense1.dinputs[1,1:10]}\n'+
                          #f'activation1.dinputs: {activation1.dinputs[1,1:5,0,0]}\n'+
                          #f'dense2.dinputs: {dense2.dinputs[1,1:5]}\n'+
                          f'truth:      {y[0:20]}\n'+
                          f'prediction: {predictions[0:20]}\n'
                          )
            
            Numpy_net =[Conv1,flatten1,dense1,activation1,dense2]
            def save_object(obj, filename):
                with open(filename, 'wb') as outp:  # Overwrites any existing file.
                    dill.dump(obj, outp)
                
            save_object(Numpy_net, 'Numpy_net_Kaggle_2.pkl')
            #you need to save with dill here because pickle save would not save attributes added after
            #object initalization!
  
            
            #fig2, ax2 = plt.subplots(3, 1,sharex=True)
            #ax2[0].plot(np.arange(Nsteps), Monitor[:,0])
            #ax2[0].set_ylabel('accuracy [%]')
            #ax2[1].plot(np.arange(Nsteps), Monitor[:,1])
            #ax2[1].set_ylabel('loss')
            #ax2[2].plot(np.arange(Nsteps), Monitor[:,2])
            #ax2[2].set_ylabel(r'$\alpha$')
            #ax2[2].set_xlabel('iteration')
            #plt.xscale('log',base=10) 
            #plt.show(fig2) #Ich will die plots schon beim ausfuehren sehen. bzw einen sich updatenden plot
       
            #with open('Numpy_net.pkl', 'rb') as inp:
            #        Conv1,flatten1,dense1,activation1,dense2 = dill.load(inp)
            #
    
    
    
    
    
    
    
    
    
    
    
    
    
    



