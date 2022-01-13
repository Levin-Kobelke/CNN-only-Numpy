import numpy as np
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