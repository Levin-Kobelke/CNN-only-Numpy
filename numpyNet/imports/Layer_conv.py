import numpy as np

 
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
                from imports.lowerimports.conv_with_padding_3 import conv_with_padding_3
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
                from imports.lowerimports.conv_with_padding_3 import conv_with_padding_3
        
                
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
            
        