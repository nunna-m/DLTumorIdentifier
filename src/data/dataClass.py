import os
import numpy as np
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import yaml

#import local libraries
from . import modalityStack
#import modalityStack
#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion() #interactive mode

'''This following class takes filenames and makes numpy equivalent on the fly'''
class KidneyTumorDataset1(Dataset):
    """Kidney Tumor image dataset two class labels AML and CCRCC"""
    def __init__(self, root_dir, modalities, foldNum, train_or_test, cropType=None, cv=None, transform=None):
        """
        Args:
            root_dir (string): root dir with all images "/../kt_combined"
            modalities (List(str)): for example ["am,"dc"]
            foldNum (int): fold number if 5CV eg: could be 0|1|2|3|4
            train_or_test (string): 'train' or 'test' to get respective subjects from fold yaml file
            cropType (string): if None then fullImage else send 'centerCrop'|'pixelCrop'
            cv (string): '5CV' or '10CV', if want LOOCV first generate those folds in foldDataFiles folder
            transform (callable, optional): Optional transform to apply on image. Can be used for augmentations.
        """
        self.root_dir = root_dir
        self.cropType = cropType
        self.cv = cv
        if cropType is None:
            self.cropType = 'fullImage'
        if cv is None:
            self.cv = '5CV'
        self.foldNum = foldNum
        self.train_or_test = train_or_test
        self.modalitiesList = sorted(modalities)
        self.modalitiesString = '_'.join(self.modalitiesList)
        self.foldDir = os.path.join(self.root_dir,self.modalitiesString,'foldDataFiles', self.cv, f'foldSubjectPaths{self.foldNum}.yaml')
        self.idxToSubjectPathMapping = dict()
        with open(self.foldDir,'r') as file:
            data = yaml.safe_load(file)[self.train_or_test]
        for i,subject_label in enumerate(data):
            self.idxToSubjectPathMapping[i] = subject_label
        self.transform = transform
    
    def __len__(self):
        #returns number of subjects in the specific fold and train|test(not number of samples)
        return len(self.idxToSubjectPathMapping)
    
    def __getitem__(self, index):
        subject_labelName = self.idxToSubjectPathMapping[index]
        subjectID, classLabel = subject_labelName.split('_')
        dataPath = os.path.join(self.root_dir, self.modalitiesString, 'rawData', classLabel, subjectID)
        subject_data = modalityStack.combineDataNew(dataPath, self.cropType)
        
        if self.transform:
            transformed_data = []
            for subject in subject_data:
                transformed_data.append(self.transform(subject))
            return transformed_data
        
        return subject_data

'''This following class takes numpy files directly (given that they're pre generated and just reads the npz files'''
class KidneyTumorDataset2(Dataset):
    """Kidney Tumor image dataset two class labels AML and CCRCC"""
    def __init__(self, root_dir, modalities, foldNum, train_or_test, cropType=None, cv=None, transform=None):
        """
        Args:
            root_dir (string): root dir with all images "/../kt_combined"
            modalities (List(str)): for example ["am,"dc"]
            foldNum (int): fold number if 5CV eg: could be 0|1|2|3|4
            train_or_test (string): 'train' or 'test' to get respective subjects from fold yaml file
            cropType (string): if None then fullImage else send 'centerCrop'|'pixelCrop'
            cv (string): '5CV' or '10CV', if want LOOCV first generate those folds in foldDataFiles folder
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

def showImage(image, label):
    '''show Image'''
    image = image[:,:,0]
    plt.imshow(image)
    if label == 0:
        heading = 'AML'
    else:
        heading = 'CCRCC'
    plt.title(heading)
    plt.pause(2)

def callThisWhenNeeded():
    '''testing out dataset generating in pytorch'''
    # kt_dataset = KidneyTumorDataset1(
    #     root_dir='/home/maanvi/LAB/Datasets/kt_combined',
    #     modalities=('am','dc'),
    #     foldNum=0,
    #     train_or_test='train',
    #     cropType='pixelCrop')

    fig = plt.figure()
    cv = 5
    foldNum = 0
    root = '/home/maanvi/LAB/Datasets/kt_combined'
    modalities = ["am","dc","ec","pc"]
    train_or_test = 'train'
    cropType = 'pixelCrop'
    kt_dataset = KidneyTumorDataset1(
            root_dir=root,
            modalities=modalities,
            foldNum=foldNum,
            train_or_test=train_or_test,
            cropType=cropType
            )
    samples = []
    for i in range(len(kt_dataset)):
        samples.extend(kt_dataset[i])

    fig = plt.figure()
    for i,sample in enumerate(samples):
        print(i, sample['image'].shape, sample['label'])
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        showImage(**sample)

        if i == 3:
            plt.show()
            break
    
class Rescale(object):
    '''Rescale the image in a sample to a given size.
    Args:
        output_size(tuple): Desired output size. eg (128,128)
    '''
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.outut_size = output_size
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h,w = image.shape[:2]
        new_h, new_w = self.output_size
        img = transform.resize(image, (new_h, new_w))
        return {'image':img, 'label':label}

class RandomCrop(object):
    '''Crop randomly the image in a sample
    Args:
        output_size (tuple or int): Desired output size if tuple eg (128,128). If int, square crop is done
    '''
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.randomint(0,)