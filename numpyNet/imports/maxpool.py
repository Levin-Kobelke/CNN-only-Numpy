import numpy as np
class maxpool:
            def __init__(self,size,stride):
                self.size=size
                self.stride=stride
            def forward(self,inputs):
                from imports.lowerimports.im2col_indices import im2col_indices
                #to copy stanford code in need to reorganize axis
                #input is (h,w,d,n) stanford(n,d,h,w)
                self.inputs=inputs
                size=self.size
                stride=self.stride
                
                X= inputs.transpose(2,3,1,0)
                self.X = X
                n,d,h,w = np.shape(X)
                # First, reshape it to 50x1x28x28 to make im2col arranges it fully in column
                X_reshaped = X.reshape(n * d, 1, h, w)
                self.X_reshaped=X_reshaped
                # The result will be 4x9800
                # Note if we apply im2col to our 5x10x28x28 input, the result won't be as nice: 40x980
                self.X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)
                
                # Next, at each possible patch location, i.e. at each column, we're taking the max index
                self.max_idx = np.argmax(self.X_col, axis=0)
                
                # Finally, we get all the max value at each column
                # The result will be 1x9800
                out = self.X_col[self.max_idx, range(self.max_idx.size)]
                h_out=int(h/size)
                w_out=int(w/size)
                # Reshape to the output size: 14x14x5x10
                out = out.reshape(h_out, w_out, n, d)
                
                # Transpose to get 14,14,10,5 output
                self.output = out.transpose(0, 1, 2, 3)
                #output is (h_out,w_out,d,n) stanford(n,d,h,w)
            def backward(self,dvalues):
                # X_col and max_idx are the intermediate variables from the forward propagation step
                from imports.lowerimports.col2im_indices import col2im_indices
                # Suppose our output from forward propagation step is 5x10x14x14
                # We want to upscale that back to 5x10x28x28, as in the forward step
                
                # 4x9800, as in the forward step
                dX_col = np.zeros_like(self.X_col)
                
                # 5x10x14x14 => 14x14x5x10, then flattened to 1x9800
                # Transpose step is necessary to get the correct arrangement
                dout_flat = dvalues.transpose(0, 1, 3, 2).ravel()
                
                # Fill the maximum index of each column with the gradient
                
                # Essentially putting each of the 9800 grads
                # to one of the 4 row in 9800 locations, one at each column
                dX_col[self.max_idx, range(self.max_idx.size)] = dout_flat
                
                # We now have the stretched matrix of 4x9800, then undo it with col2im operation
                # dX would be 50x1x28x28
                h,w,d,n=np.shape(self.inputs)
                dX = col2im_indices(dX_col, (n * d, 1, h, w), self.size, self.size, padding=0, stride=self.stride)
                
                # Reshape back to match the input dimension: 5x10x28x28
                dX = dX.reshape(self.X.shape)
                self.dinputs = dX.transpose(2,3,0,1)
                