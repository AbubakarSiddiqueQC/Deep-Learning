import numpy as np
import matplotlib.pyplot as plt
from neural_network import *
from f_load_dataset import load_dataset

# load dataset
train_x, train_t,test_x,test_t = load_dataset()
#idx = 0
#minibatch_input =  train_x[idx:idx + 32,:]
#minibatch_target =  train_t[idx:idx + 32,:]


# create l-dim network by just adding num of neurons in layer_dim
# first and last elements represent input and output layers dim
layer_dim = [784,5,5,2]
keep_prob = [1,0.8,0.8,1]
# add activation functions name here. 
# input layer activation function is None
activations = [None, 'sigmoid','sigmoid','softmax']
assert len(layer_dim) ==  len(activations), "layer dim or activation is missing.."

# hyper parameters of neural network
learning_rate = 1e-3
num_epochs = 100
mini_batch_size = 32


nn = NeuralNetwork(layer_dim, activations, learning_rate, num_epochs, mini_batch_size)

# train neural network 
nn.train(train_x, train_t,keep_prob)
Acc,all_count = nn.test(test_x,test_t)
print("Number Of Images Tested =", all_count)
print("\nTesting Accuracy =", Acc)
Acc,all_count = nn.test(train_x,train_t)
print("Number Of Images Tested =", all_count)
print("\nTraining Accuracy =", Acc)






