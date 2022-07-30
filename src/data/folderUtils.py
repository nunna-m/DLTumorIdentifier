import os
import yaml
import shutil
from . import modalityStack

def createRawDataFolder(oldPath, newPath):
    '''copy data as is from kt_trainvaltest into kt_combined
    Args:
        oldPath: path of kt_trainvaltest
        newPath: resultant path to store copied data
    '''
    trainvaltest = ['train','val','test']
    classes = ['AML','CCRCC']
    runningFolders = []
    for modFolder in os.listdir(oldPath):
        if not os.path.exists(os.path.join(newPath,modFolder)):
            runningFolders.append(modFolder)
            os.makedirs(os.path.join(newPath,modFolder),exist_ok=True)
            os.makedirs(os.path.join(newPath,modFolder,'rawData'),exist_ok=True)
            for clas in classes:
                os.makedirs(os.path.join(newPath,modFolder,'rawData',clas),exist_ok=True)
                for subfold in ['train','test']:
                    os.makedirs(os.path.join(newPath,modFolder,'numpyData','fullImage',subfold),exist_ok=True)
                    os.makedirs(os.path.join(newPath,modFolder,'numpyData','centerCrop',subfold),exist_ok=True)
                    os.makedirs(os.path.join(newPath,modFolder,'numpyData','pixelCrop',subfold),exist_ok=True)

    for modFolder in runningFolders:
        for splitType in trainvaltest:
            for clas in classes:
                source = os.path.join(oldPath,modFolder,splitType,clas)
                dest = os.path.join(newPath,modFolder,'rawData',clas)
                for subjectID in os.listdir(source):
                    shutil.copytree(os.path.join(source,subjectID),os.path.join(dest,subjectID))
    

def createNPZFiles(basePath,trainSubjects,testSubjects,foldType, foldNum):
    '''create numpy z files from subject data dict returned by modalityStack combine data function
    Args:
        basePath: path of modality, child folders would be rawData, numpyData and foldData
        trainSubjects: 1234_CCRCC is example of subject stored in folds
        testSubjects: similar to train
        foldType: 5CV or 10CV
        foldNum: based on range (0,foldTypeValue (either 5 or 10))
    '''
    rawDataPath = os.path.join(basePath,'rawData')
    newDataPath = os.path.join(basePath,'numpyData')

    for filename in trainSubjects:
        subjectID, clas = filename.split('_')
        source = os.path.join(rawDataPath,clas,subjectID)
        for typ in ['fullImage','centerCrop','pixelCrop']:
            dest = os.path.join(newDataPath,typ,'train',foldType,foldNum)
            os.makedirs(dest,exist_ok=True)
            arr = modalityStack.combineData(source,dest,typ)
            #print(f'checking image shape: {arr.shape}')
    
    for filename in testSubjects:
        subjectID, clas = filename.split('_')
        source = os.path.join(rawDataPath,clas,subjectID)
        for typ in ['fullImage','centerCrop','pixelCrop']:
            dest = os.path.join(newDataPath,typ,'test',foldType,foldNum)
            os.makedirs(dest,exist_ok=True)
            arr = modalityStack.combineData(source,dest,typ)
            #print(f'checking image shape: {arr.shape}')


def createNumpyFiles(oldPath, newPath):
    '''calling function for createNPZfiles by looping over train and test subjects in respective foldData. Called after creating folds files
    Args:
        oldPath: path of kt_trainvaltest
        newPath: resultant path to store copied data
    '''
    for modFolder in os.listdir(oldPath):
        print(f"creating numpy files of {modFolder}")
        for folder in ['5CV','10CV']:
            foldsPath = os.path.join(newPath,modFolder,'foldDataFiles',folder)
            for foldD in os.listdir(foldsPath):
                foldNum = os.path.splitext(foldD)[0][-1]
                with open(os.path.join(foldsPath,foldD),'r') as file:
                    data = yaml.safe_load(file)
                createNPZFiles(os.path.join(newPath,modFolder),data['train'],data['test'],folder, foldNum)
