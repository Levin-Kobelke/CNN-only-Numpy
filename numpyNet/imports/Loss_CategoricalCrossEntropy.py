import numpy as np
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
