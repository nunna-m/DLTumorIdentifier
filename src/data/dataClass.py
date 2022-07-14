from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import yaml
#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion() #interactive mode

class KidneyTumorAMLCCRCC(Dataset):
    """Kidney Tumor image dataset two classes AML and CCRCC"""
    def __init__(self, paths_file, root_dir, modalities, training=True, transform=None):
        """
        Args:
            paths_file (string): file path for training and testing subjects in that fold
            root_dir (string): root dir with all modalities' directory which inturn have crossval fold files and train val test old folder images. basically /.../kt_new_trainvaltest
            transform (callable, optional): Optional transform to apply on image. Can be used for augmentations.
        """
        with open(paths_file,'r') as file:
            if training:
                self.subject_paths = yaml.safe_load(file)['train']
            else:
                self.subject_paths = yaml.safe_load(file)['test']
        self.modalities_combined_root_dir = os.path.join(root_dir, '_'.join(modalities))
        self.modalities = modalities
        self.training = training
        self.transform = transform
    
    def __len__(self):
        #call the folders.py function to get counts
        #for now return length of yaml file number of train/test subjects
        return len(self.subject_paths)
    
    def __getitem__(self, idx):
        '''
        inner function to stack different labels of each modality for every subject
        if number of modalities is 1, create a copy of same image 3 times to make 3D image
        if number of modalities is 2, concatenate with empty array
        if number of modalities >= 3, use as is

        take directory of subject and return decoded images stack them according to image sample number for each modality
        '''
        #if idx is tensor format convert to list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #combine images from different modalities logic
        subject_path = self.subject_paths[idx]


        
        