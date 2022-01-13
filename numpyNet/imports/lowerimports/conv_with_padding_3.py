# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 15:05:50 2021

@author: Levin user
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 14:41:11 2021

@author: Levin user
"""
  
def conv_with_padding_3(a, f):
    import numpy as np
    #   """
    #   keyword arguments
    #   a: input image
    #   f: filter to be applied filter
    #   calculates convolution of image with one filter
    #   """
    filter_size = f.shape[1] #filter row&col length
    a = np.pad(a,int(filter_size/2),mode = 'constant',constant_values =0)
    #padding of the image will result in increased image but cropped due to conv
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape = s, strides = a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)