import h5py
import numpy as np


# 500 samples for 100 per class
def load_test_data(positive=0):
    '''
    package datasets according to positive

    :param positive: class [positive] belong to positive sample and others to be negative
    :return: train_X, train_Y
    '''
    datasets = h5py.File("../carDatasets.h5")
    test_X = np.array(datasets['test_X'])
    test_Y = np.array(datasets['test_Y'])
    datasets.close()
    test_Y[0][100*positive:100*positive+100]=1
    test_Y[0][:100*positive]=test_Y[0][100*positive+100:500]=0
    return test_X, test_Y


## 5 classes of train_data  200 per class
def load_train_data(positive=0):
    '''
    package datasets according to positive

    :param positive: class [positive] belong to positive sample and others to be negative
    :return: train_X, train_Y
    '''
    datasets = h5py.File("../carDatasets.h5")
    train_X = np.array(datasets['train_X'])
    train_Y = np.array(datasets['train_Y'])
    datasets.close()
    train_Y[0][200*positive:200*positive+200]=1
    train_Y[0][:200*positive]=train_Y[0][200*positive+200:1000]=0
    return train_X, train_Y
