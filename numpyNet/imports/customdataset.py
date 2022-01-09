# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 12:12:59 2022

@author: Levin_user
Custum Dataloader Class Training images cats vs dogs
"""

import os
import pandas as pd
from torchvision.io import read_image
import torch
import torchvision.transforms as transforms

class CustomImageDataset():
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = torch.tensor(label, dtype=torch.int64)
            label = self.target_transform(label)
        return image, label
