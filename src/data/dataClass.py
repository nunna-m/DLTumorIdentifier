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
    def __init__(self, root_dir, modalities, foldNum, train_or_test, cropType=None, outputsize=(224,224),cv=None, transform=None):
        """
        Args:
            root_dir (string): root dir with all images "/../kt_combined"
            modalities (List(str)): for example ["am,"dc"]
            foldNum (int): fold number if 5CV eg: could be 0|1|2|3|4
            train_or_test (string): 'train' or 'test' to get respective subjects from fold yaml file
            cropType (string): if None then fullImage else send 'centerCrop'|'pixelCrop'
            outputsize (tuple): (resultant_width,resultant_height)
            cv (string): '5CV' or '10CV', if want LOOCV first generate those folds in foldDataFiles folder
            transform (callable, optional): Optional transform to apply on image. Can be used for augmentations.
        """
        self.root_dir = root_dir
        self.cropType = cropType
        self.outputsize = outputsize
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
        subject_data = modalityStack.combineDataNew(dataPath, self.cropType, self.outputsize)
        
        if self.transform:
            transformed_data = []
            for subject in subject_data:
                transformed_data.append(self.transform(subject))
            return transformed_data
        
        return subject_data


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
    fig = plt.figure()
    cv = 5
    foldNum = 0
    root = '/home/maanvi/LAB/Datasets/kt_combined'
    modalities = ["am","dc","ec","pc"]
    train_or_test = 'train'
    #cropTypeMapping = {'centerCrop':'center','pixelCrop':'pixel','fullImage': None}
    cropType = 'fullImage'
    outputsize = (200, 400)
    kt_dataset = KidneyTumorDataset1(
            root_dir=root,
            modalities=modalities,
            foldNum=foldNum,
            train_or_test=train_or_test,
            cropType=cropType,
            outputsize=outputsize,
            )
    samples = []
    for i in range(len(kt_dataset)):
        samples.extend(kt_dataset[i])

    scale = Rescale(256)
    crop = RandomCrop(128)
    composed = transforms.Compose([Rescale(256),RandomCrop(224)])
    fig = plt.figure()
    #simple plotting of images without transforms
    # for i,sample in enumerate(samples):
    #     print(i, sample['image'].shape, sample['label'])
    #     ax = plt.subplot(1, 4, i + 1)
    #     plt.tight_layout()
    #     ax.set_title('Sample #{}'.format(i))
    #     ax.axis('off')
    #     showImage(**sample)

    #     if i == 3:
    #         plt.show()
    #         break

    #plotting transformed samples
    #use samples instead of kt_dataset because samples is flattened version of kt
    sample = samples[42]
    for i, tsfrm in enumerate([scale, crop, composed]):
        transformed_sample = tsfrm(sample)
        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        showImage(**transformed_sample)
    
    plt.show()

class RandomCrop(object):
    '''Crop randomly the image in a sample
    Args:
        output_size (tuple or int): Desired output size if tuple eg (128,128). If int, square crop is done
    '''
    def __init__(self, outputsize):
        assert isinstance(outputsize, (int, tuple))
        if isinstance(outputsize, int):
            self.output_size = (outputsize, outputsize)
        else:
            assert len(outputsize) == 2
            self.output_size = outputsize
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = torch.randint(0, h - new_h, (1,))
        left = torch.randint(0, w - new_w, (1,))

        image = image[top: top + new_h,
                      left: left + new_w]
        
        return {'image': image, 'label':label}
    
class ToTensor(object):
    '''Convert ndarray in sample to Tensors'''
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        #swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2,0,1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'label':label}

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
