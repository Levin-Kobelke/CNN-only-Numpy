class Activation_Softmax_Loss_CategoricalCrossentropy():
    
    def __init__(self):
        from Activation_Softmax import Activation_Softmax
        from Loss_CategoricalCrossEntropy import Loss_CategoricalCrossEntropy
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
        