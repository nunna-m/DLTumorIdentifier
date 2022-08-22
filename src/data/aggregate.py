import os
from . import crossValFolds
from . import folderUtils

def createRaw(oldPath, newPath):
    '''copy kt_new_trainvaltest to kt_combined. Run only once.
    Args:
        oldPath: path to kt_new_trainvaltest where each data for each modality is there separately
        newPath: creating a copy of old data in new folder kt_combined just in case
    '''
    os.makedirs(newPath, exist_ok=True)
    folderUtils.createRawDataFolder(oldPath=oldPath, newPath=newPath)

def createNumpy(oldPath, newPath, numpyFiles=False):
    '''create numpy files for every modalitiy, subject, and 5CV and 10CV train and test separately. Lot of repetitive data stored -> Sample command: python -m src.data createNumpy --oldPath "D:\01_Maanvi\LABB\datasets\kt_new_trainvaltest" --newPath "D:\01_Maanvi\LABB\datasets\kt_combined" [--numpyFiles] include last flag if you want numpy files
    Args:
        oldPath: path to kt_new_trainvaltest where each data for each modality is there separately
        newPath: path to kt_combined
        numpyFiles (bool): if False, just create folds, if true create numpy files also
    '''
    for modFolder in os.listdir(oldPath):
        path = os.path.join(newPath,modFolder,'rawData')
        crossValFolds.createFolds(basePath=path)
        print(f"done creating {modFolder} modality folds")
    if numpyFiles:
        print("Create numpy files")
        folderUtils.createNumpyFiles(oldPath=oldPath, newPath=newPath)


#windows sample command
# python -m src.data createRaw --oldPath "D:\01_Maanvi\LABB\datasets\kt_new_trainvaltest" --newPath "D:\01_Maanvi\LABB\datasets\kt_combined"

#linux sample command
# python3 -m src.data createRaw --oldPath "/home/maanvi/LAB/Datasets/kt_new_trainvaltest" --newPath "/home/maanvi/LAB/Datasets/kt_combined"

#remote sample command
# python3 -m src.data createRaw --oldPath "/kw_resources/datasets/kt_new_trainvaltest" --newPath "/kw_resources/datasets/kt_combined"

#similarly for createNumpy