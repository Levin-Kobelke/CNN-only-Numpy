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