 class Activation_Sigmoid:
            
            def forward(self, inputs):
                self.output  = 1/(1 + np.exp(-inputs-np.max(inputs,axis =1,\
                                                            keepdims = True)))
                #max in order to prevent overflow
                
                #sigmoid funtion
                self.inputs  = inputs
                
            def backward(self,dvalues):
                self.dinputs = self.output * (1-self.output)
                #derivative of a sigmoid