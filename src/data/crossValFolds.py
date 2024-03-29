from operator import mod
import os
import random
import yaml
import glob
from sklearn.model_selection import KFold
import numpy as np

def createFolds(basePath, kfolds=None):
    '''using basePath create fold data with subjects and respective classes
    Args:
        basePath: path of modality folder + rawData whose children are AML and CCRCC
        kfolds: if None, create both 5CV and 10CV, however if int like 5 or 8 is passed that CV is created so 8CV
    '''
    paths = {'AML':list(), 'CCRCC':list()}
    all_classes_path = glob.glob(basePath+'/*/*')
    target_classes = []
    for subject in all_classes_path:
        subjectCopy = subject[:]
        clas = subjectCopy.rsplit(os.path.sep,2)[1]#AML or CCRCC
        paths[clas].append(subject)
        target_classes.append(clas)
    
    parentPath = basePath.rsplit(os.path.sep,1)[0]
    foldsPath = os.path.join(parentPath,'foldDataFiles')
    
    #create folds folder if doesn't exist
    os.makedirs(foldsPath,exist_ok=True)

    with open(os.path.join(foldsPath,'allSubjectPaths.yaml'),'w') as file:
        yaml.dump(paths,file)
    
    #kfolds CV
    if kfolds is None:
        #kfolds = [5, 10, len(paths['AML'])] #use if want LOOCV also
        kfolds = [5, 10]
    else:
        kfolds = [kfolds]
    for kfold in kfolds:
        k = KFold(n_splits=kfold, shuffle=False)
        train_full = {i:list() for i in range(kfold)}
        test_full = {i:list() for i in range(kfold)}
        for clas in set(target_classes):
            fold_num = 0
            for train_index, test_index in k.split(paths[clas]):
                train_full[fold_num].extend(np.take(paths[clas],train_index))
                test_full[fold_num].extend(np.take(paths[clas],test_index))
                fold_num += 1
        if kfold > 10:
            storeCVPath = os.path.join(foldsPath,'LOOCV')
        else:
            storeCVPath = os.path.join(foldsPath,f'{kfold}CV')
        os.makedirs(storeCVPath, exist_ok=True)
        for i in range(kfold):
            random.shuffle(train_full[i])
            random.shuffle(test_full[i])
            new_train_full = [f'{path.rsplit(os.path.sep,2)[2]}_{path.rsplit(os.path.sep,2)[1]}' for path in train_full[i]]
            new_test_full =  [f'{path.rsplit(os.path.sep,2)[2]}_{path.rsplit(os.path.sep,2)[1]}' for path in test_full[i]]
            store = {'train':new_train_full,'test':new_test_full}
            with open(os.path.join(storeCVPath,f'foldSubjectPaths{i}.yaml'),'w') as file:
                yaml.dump(store,file)


def generateExtra4ModalFolds(path, modalities):
    '''
    Outer function to call createFolds on newly created 4 set modalities
    Sample command to call: python3 -m src.data generateExtra4ModalFolds --ktcombipath "/../path" --modalities am dc ec pc
    Args:
        path: kt combi path '/home/maanvi/LAB/datasets/kt_combined'
        modalities (list[str]): newly added modalities for which fold data must be added
    '''
    modalitiesString = '_'.join(modalities)
    newPath = os.path.join(path, modalitiesString, 'rawData')
    createFolds(newPath)

# createFolds('/home/maanvi/LAB/Datasets/kt_combined/am_ec_tm/rawData')
# createFolds('/home/maanvi/LAB/Datasets/kt_combined/ec_tm/rawData')
# createFolds('/home/maanvi/LAB/Datasets/kt_combined/pc/rawData')

#'/home/maanvi/LAB/Datasets/kt_combined/pc/rawData/AML/15630104'