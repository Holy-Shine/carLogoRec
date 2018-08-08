import h5py
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (3.0, 2.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


np.random.seed(1)

def showImage(index,dataset=None,label='train_X'):
    names = ['Citroen','Volkswagen','Faw','Fonton','Honda']
    assert dataset!=None,'Please make sure you have load the correct dataset!'
    assert label=='train_X' or label=='test_X','Please make sure label=\'train_X\'or \'test_X\'!'
    image_arr = np.array(dataset[label][:])
    image_arr=image_arr.reshape(-1,32,32)
    plt.imshow(image_arr[index,:,:])
    plt.title(names[int(dataset[label[:-1]+'Y'][index][0])])
    plt.show()


dataset=h5py.File('carDatasets.h5','r')
showImage(900,dataset)