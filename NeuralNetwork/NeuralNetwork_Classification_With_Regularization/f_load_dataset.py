import numpy as np
import matplotlib.pyplot as plt
import numpy as np




def load_dataset():   
    image_size = 28 # width and length
    no_of_different_labels = 2 #  i.e. 0, 1, 2, 3, ..., 9
    image_pixels = image_size * image_size
    data_path = "data/mnist/"
    train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")
    test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",") 
    #print(train_data.shape)
    #print(test_data.shape)
    
    fac = 0.99 / 255
    train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
    test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01
    #print(train_imgs.shape)
    #print(test_imgs.shape)
    train_labels = np.asfarray(train_data[:, :1])
    test_labels = np.asfarray(test_data[:, :1])
    #print(train_labels.shape)
    #print(test_labels.shape)
    train_filter = np.array(np.where((train_labels == 1 ) | (train_labels == 2)))
    test_filter = np.array(np.where((test_labels == 1) | (test_labels == 2)))
    #print(train_filter.shape)
    #print(test_filter.shape)
    train_imgs_bin = np.zeros([train_filter.shape[1],image_pixels])
    train_labels_bin = np.zeros([train_filter.shape[1],1])
    test_imgs_bin = np.zeros([test_filter.shape[1],image_pixels])
    test_labels_bin = np.zeros([test_filter.shape[1],1])
    for n in range(train_filter.shape[1]):
        train_imgs_bin[n] = train_imgs[train_filter[0,n]]
        train_labels_bin[n] = train_labels[train_filter[0,n]]
    for m in range(test_filter.shape[1]):
        test_imgs_bin[m] = test_imgs[test_filter[0,m]]
        test_labels_bin[m] = test_labels[test_filter[0,m]]
    
    lr = np.arange(1,no_of_different_labels+1)

    # transform labels into one hot representation
    train_labels_bin_one_hot = (lr==train_labels_bin).astype(np.float)
    test_labels_bin_one_hot = (lr==test_labels_bin).astype(np.float)

    # we don't want zeroes and ones in the labels neither:
    train_labels_bin_one_hot[train_labels_bin_one_hot==0] = 0.01
    train_labels_bin_one_hot[train_labels_bin_one_hot==1] = 0.99
    test_labels_bin_one_hot[test_labels_bin_one_hot==0] = 0.01
    test_labels_bin_one_hot[test_labels_bin_one_hot==1] = 0.99
    
    return train_imgs_bin,train_labels_bin_one_hot,test_imgs_bin,test_labels_bin_one_hot
   