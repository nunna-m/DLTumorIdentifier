import os
from . import newCrossVal
from . import newFolders

datasets_root = '/home/maanvi/LAB/Datasets'
#datasets_root = '/kw_resources/datasets'
oldPath = os.path.join(datasets_root,'kt_new_trainvaltest')
newPath = os.path.join(datasets_root,'kt_combined')
os.makedirs(newPath, exist_ok=True)

def createRaw(oldPath, newPath):
    '''copy kt_new_trainvaltest to kt_combined. Run only once.
    Args:
        oldPath: path to kt_new_trainvaltest where each data for each modality is there separately
        newPath: creating a copy of old data in new folder kt_combined just in case
    '''
    newFolders.createRawDataFolder(oldPath=oldPath, newPath=newPath)

def createNumpy(oldPath, newPath):
    '''create numpy files for every modalitiy, subject, and 5CV and 10CV train and test separately. Lot of repetitive data stored
    Args:
        oldPath: path to kt_new_trainvaltest where each data for each modality is there separately
        newPath: path to kt_combined
    '''
    for modFolder in os.listdir(oldPath):
        path = os.path.join(newPath,modFolder,'rawData')
        newCrossVal.createFolds(basePath=path)
        print(f"done creating {modFolder} modality folds")
    newFolders.createNumpyFiles(oldPath=oldPath, newPath=newPath)

    