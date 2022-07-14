import os
import torch
import cv2
import matplotlib.pyplot as plt
import yaml

def train_ds(
    data_root,
    modalities,
    batch_size,
    buffer_size,
    repeat=True,
    output_size=(224,224),
    aug_configs=None,
    tumor_region_only=False,
):
    '''
    create train dataset
    '''
    with open(data_root, 'r') as file:
        data = yaml.safe_load(file)
    traindir = data['train']
    dataset = load(
        traindir,
        modalities=modalities,
        output_size=output_size,
        tumor_region_only=tumor_region_only,
    )
    dataset = configure_dataset(
        dataset,
        batch_size,
        buffer_size,
        repeat=repeat,
    )
    print(f"Final Dataset: {dataset}")
    return dataset

def load(traindir,
        modalities=('am','dc','ec','pc','tm'),
        output_size=(224,224),
        tumor_region_only=False,
):
    '''
    generate the base dataset
    '''
    trainSubjectPaths = traindir

def configure_dataset(ds,batch_size=32,buffer_size=1024,repeat=True):
    pass