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
        
                
           
        #definint the flattening method to input convolution into dense layers
        
      
        
        #we now defines the layers from the previous chapter as classes and can 
        #therefore define weights and biases as method
        #the fully connected (fc) layer we had before is also called "dense layer"
        
       
                
       
            
     
            
            
         #defining a common loss function that just calls the part loss function
         #and calculates the mean over all samples
        class Loss:
             
             def calculate(self, output, y):
                 
                 sample_losses = self.forward(output, y)
                 data_loss     = np.mean(sample_losses)
                 return(data_loss)
            
            

                 

                

                
                
                
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    



