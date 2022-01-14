import numpy as np
class Flatten:
        def forward(self, inputs):
            self.output = inputs.reshape(-1,inputs.shape[-1]).T#flattenes array into 1D in cols for dense
            
            self.inputs = inputs
        def backward(self, dvalues):
            nr,nc,nd,nI = np.shape(self.inputs)
            dvalues = dvalues.T
            self.dinputs = dvalues.reshape((nr,nc,nd,nI))
            