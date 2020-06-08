import numpy as np
import matplotlib.pyplot as plt
from network import Network
from f_data_loader import *


shuffle_data = True
batch_size = 32
val_batch_size = 32
epoch = 3
train_x, train_y, val_x, val_y, test_x, test_y = load_dataset_mnist(shuffle_data)


net = Network(batch_size, val_batch_size, epoch)
print('train CNN...')
net.train(train_x, train_y, val_x, val_y)
print('test CNN....')
net.test(test_x, test_y)

