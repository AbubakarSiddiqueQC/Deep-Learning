
import numpy as np
import matplotlib.pyplot as plt
from neural_network import *
from f_load_dataset import load_dataset
# load dataset
batch_size = 100
train_loader, test_loader = load_dataset(batch_size)

# create l-dim network by just adding num of neurons in layer_dim
# first and last elements represent input and output layers dim
layer_dim = [784, 50,50, 10]

# add activation functions name here. 
# input layer activation function is None
activations = [None, 'relu','relu', 'identity']
assert len(layer_dim) ==  len(activations), "layer dim or activation is missing.."

# hyper parameters of neural network
learning_rate = 1e-1
num_epochs = 15
#mini_batch_size = 10

# train neural network 
train(train_loader, learning_rate, num_epochs,layer_dim, activations)


# test neural network 
test(test_loader)

