from PIL import Image
import numpy as np
import h5py
import re,os

pattern = ".+\.jpg"
trainPaths = [
    "./Cardata/train/0",
    "./Cardata/train/1",
    "./Cardata/train/2",
    "./Cardata/train/3",
    "./Cardata/train/4"
    ]
testPaths = [
    "./Cardata/test/0",
    "./Cardata/test/1",
    "./Cardata/test/2",
    "./Cardata/test/3",
    "./Cardata/test/4"
]




def create_dataset(size=32):
    '''
    Model the datasets

    train_X: (n_features, n_samples)
    train_Y: (1, n_samples)
    ...
    :return: None
    '''
    dataset = h5py.File('carDatasets.h5', 'w')
    # train data
    ## sample 1000
    ## size: resize to size x size 
    train_X = np.zeros(shape=(1000, size*size),dtype=float)
    train_Y = np.zeros(shape=(1000,1),dtype=float)
    count = 0
    for trainpath in trainPaths:
        for img in [file for file in os.listdir(trainpath) if re.match(pattern,file)]:
            img_path = trainpath+os.sep+img
            img_obj = Image.open(img_path).resize((size,size)).convert('L')
            img_array = np.asarray(img_obj).reshape((1,size*size))
            train_X[count]=img_array
            train_Y[count]=int(trainpath[-1])
            count+=1
    dataset.create_dataset('train_X', data=train_X)
    dataset.create_dataset('train_Y', data=train_Y)

    # test data
    ## sample 500
    ## size: resize to sizexsize=2025
    test_X = np.zeros(shape=(500, size*size),dtype=float)
    test_Y = np.zeros(shape=(500, 1),dtype=float)
    count = 0
    for testpath in testPaths:
        for img in [file for file in os.listdir(testpath) if re.match(pattern,file)]:
            img_path = testpath+os.sep+img
            img_obj = Image.open(img_path).resize((size,size)).convert('L')
            img_array = np.asarray(img_obj).reshape((1,size*size))
            test_X[count]=img_array
            test_Y[count]=int(testpath[-1])
            count+=1
    dataset.create_dataset('test_X', data=test_X)
    dataset.create_dataset('test_Y', data=test_Y)
    dataset.close()

if __name__=='__main__':
    create_dataset()
    f = h5py.File('carDatasets.h5','r')
    train_X = np.array(f['test_Y'][:])
    print(train_X)