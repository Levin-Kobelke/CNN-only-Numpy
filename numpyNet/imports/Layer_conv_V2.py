import numpy as np

 
class Layer_conv:
            """
            Should take as input data from the previous layer in format H,W,C,N height, width, channel, Number
            and return H,W,C,N where H,W is dependent on the filter size and step size, but are equal to the inputs for
            padding = 1, Stride = 1 and filter size = 3 (most common)
            Stepszide and padding is always 1 for simplizity of this project
            C is the number of different filters used for convolution and N as image indicater does not change too
            filters are randomly innitiated

            This is the most likly file to contain mathematical errors. Please read carefully if there are any mistakes in computing the backwards part!
            """
                
            def __init__(self, n_neurons = 1,input_channels = 3,size = 3):
                """
                Initalizes convolution layer
        
                Parameters
                ----------
                n_neurons : int, optional
                    How many different classes of neurons(e.g. filters) to create. 
                input_channels: int
                    Number of channels the input has.
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
                ( n_H, n_W, n_C, m)=dvalues.shape
                #write into a matrix and done
                weights = self.weights
                (f,f,n_C_prev,n_C) = weights.shape
                (n_H_prev,n_W_prev,n_C_prev,m) = self.inputs.shape

                #initalize with correct shapes
                dweights = np.zeros((f,f,n_C_prev,n_C))
                dbias = np.zeros((1,1,1,n_C))
                dinputs = np.zeros((n_H_prev,n_W_prev, n_C_prev,m))

                # npad is a tuple of (n_before, n_after) for each dimension



                npad = ((int(self.filter_size/2), int(self.filter_size/2)), (int(self.filter_size/2), int(self.filter_size/2)), (0, 0),(0,0))
                
                inp_padded =np.pad(self.inputs,pad_width=npad,mode = 'constant',constant_values =0)
                dinputs_padded =np.pad(dinputs,pad_width=npad,mode = 'constant',constant_values =0)
                for i in range(m):
                    # select ith training example from A_prev_pad and dA_prev_pad

                    inputImg = inp_padded[:,:,:,i]
                    dinputImg = dinputs_padded[:,:,:,i]
                    for h in range(n_H):                   # loop over vertical axis of the output volume
                        for w in range(n_W):               # loop over horizontal axis of the output volume
                            for c in range(n_C):           # loop over the channels of the output volume

                                # Find the corners of the current "slice"
                                row_start = h
                                row_end = row_start + f
                                col_start = w
                                col_end = col_start + f

                                #getting the slice
                                inp_slice = inputImg[row_start:row_end,col_start:col_end]

                                #updating the gradients using the math derivatives
                                dinputImg[row_start:row_end,col_start:col_end,:]+= weights[:,:,:,c]*dvalues[h,w,c,i]
                                dweights[:,:,:,c] += inp_slice * dvalues[h, w, c,i]
                                dbias[:,:,:,c] += dvalues[h, w, c,i]
                    
                    # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
                    pad = int(f/2)
                    dinputs[:, :, :,i] = dinputImg[pad:-pad, pad:-pad, :]
                self.dweights = dweights
                self.dinputs = dinputs
                self.dbias = dbias