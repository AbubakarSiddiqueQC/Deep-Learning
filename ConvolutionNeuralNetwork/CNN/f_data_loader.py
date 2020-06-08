import numpy as np
import scipy.io
from sklearn.utils import shuffle

def normalize_data(data):      
    normalized_data = np.zeros((data.shape))
    for i in range(data.shape[0]):
         img = data[i,:] 
         image = (img - np.min(img))/(np.ptp(img))
         normalized_data[i,:] =  image
    return normalized_data

def load_dataset_mnist(shuffle_data):
    data_file = scipy.io.loadmat('mnist_uint8.mat')
    train_x = data_file['train_x']
    train_y = data_file['train_y']
    test_x = data_file['test_x']
    test_y = data_file['test_y']

    if shuffle_data==True:
        train_x, train_y = shuffle(train_x, train_y, random_state=0)
    
    val_x = train_x[50000:60000,:]
    val_y = train_y[50000:60000,:]
    train_x = train_x[0:50000,:]
    train_y = train_y[0:50000,:]
    
    
    train_x = normalize_data(train_x)   
    val_x = normalize_data(val_x)
    test_x = normalize_data(test_x)
    
    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
    val_x = val_x.reshape(val_x.shape[0], 28, 28, 1)
    test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)    
    #return train_x[0:2000,:,:,:], train_y[0:2000, :], val_x[0:2000,:,:,:], val_y[0:2000,:], test_x[0:2000,:,:,:], test_y[0:2000,:]
    return train_x, train_y, val_x, val_y, test_x, test_y

