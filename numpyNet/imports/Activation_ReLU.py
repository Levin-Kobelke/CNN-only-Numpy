import numpy as np
class Activation_ReLU:
        
        def forward(self, inputs):
            self.output  = np.maximum(0,inputs)
            self.inputs  = inputs
            
        def backward(self, dvalues):
            self.dinputs = dvalues.copy()
            self.dinputs[self.inputs <= 0] = 0#ReLU derivative