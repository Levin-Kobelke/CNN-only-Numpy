#defining a common loss function that just calls the part loss function
#and calculates the mean over all samples
class Loss:
    
    def calculate(self, output, y):
        
        sample_losses = self.forward(output, y)
        data_loss     = np.mean(sample_losses)
        return(data_loss)