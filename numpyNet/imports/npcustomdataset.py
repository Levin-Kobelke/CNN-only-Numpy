# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 12:12:59 2022

@author: Levin_user
Custum Dataloader Class Training images cats vs dogs
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class CustomImageDataLoader():
    """
    this class takes a file with the classes and file names (annotiations_file.csv), the file directory and a batchsize
    It then generates an iteratble object that is able to return minibatches of the images described by there filename and directory
    """
    def __init__(self, annotations_file, img_dir,batchsize, shuffle = True):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.batchsize = batchsize
        self.maxbatches = len(self.img_labels)//self.batchsize

        self.counter = 0
        if shuffle:
            self.img_labels = self.img_labels.reindex(np.random.permutation(self.img_labels.index))


    def __len__(self):
        return len(self.img_labels)

    
    def __next__(self):
        if(self.maxbatches<=self.counter):
            raise StopIteration

        img_batch=[]
        for image in range(0,self.batchsize):
            img =[plt.imread(os.path.join(self.img_dir, self.img_labels.iloc[image+self.counter*self.batchsize, 0]))]
            img_batch.append(img)
        img_batch=np.squeeze(np.stack(img_batch,axis=4))

        label_batch = self.img_labels.iloc[self.counter*self.batchsize:(1+self.counter)*self.batchsize, 1]
        self.counter+=1

        return img_batch, label_batch

    def __iter__(self):
        return self

