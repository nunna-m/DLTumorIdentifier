from __future__ import print_function, division
import os
import numpy as np
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from xgboost import train
import yaml
#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion() #interactive mode

class KidneyTumorDataset(Dataset):
    """Kidney Tumor image dataset two class labels AML and CCRCC"""
    def __init__(self, root_dir, modalities, foldNum, train_or_test, cropType=None, cv=None, transform=None):
        """
        Args:
            root_dir (string): root dir with all images "/../kt_combined"
            modalities (List(str)): for example ["am,"dc"]
            transform (callable, optional): Optional transform to apply on image. Can be used for augmentations.
        """
        if cropType is None:
            self.cropType = 'fullImage'
        if cv is None:
            self.cv = '5CV'
        self.foldNum = foldNum
        self.train_or_test = train_or_test
        self.imageDir = os.path.join(root_dir,'_'.join(sorted(modalities)),'numpyData',cropType,train_or_test,cv,foldNum)
        self.modalities = modalities
        self.transform = transform
    def __len__(self):
        return len(os.listdir(self.imageDir))
    
    def __getitem__(self, imageName):
        imagePath = os.path.join(self.imageDir, imageName)
        data = np.load(imagePath)
        print(list(data.keys()))
        #print(f"image: {data['image']} and label:{data['label']}")
        sample = {'image': data['image'], 'label':data['label']}
        # image = data['image']
        # label = data['label']
        if self.transform:
            sample = self.transform(sample)
        return sample

kt_dataset = KidneyTumorDataset(root_dir='/home/maanvi/LAB/Datasets/kt_combined',modalities=('am','dc'),foldNum=0,train_or_test='train',cropType='pixelCrop')

fig = plt.figure()


        
        