import numpy as np
from f_utils import *
from layers import *

def define_model(self, layers):     
    self.layers.append(Convolution(inputs_channel=1, num_filters=6, kernel_size=3, padding=1, stride=1, name='conv', num=1))
    self.layers.append(ReLu(num=2))
    self.layers.append(Maxpooling(pool_size=2, stride=2, num=3))
    self.layers.append(Convolution(inputs_channel=6, num_filters=12, kernel_size=3, padding=1, stride=1, name='conv', num=4))
    self.layers.append(ReLu(num=5))
    self.layers.append(Maxpooling(pool_size=2, stride=2, num=6))
    self.layers.append(Flatten(num=7))
    self.layers.append(FullyConnected(num_inputs=7*7*12, num_outputs=10, name='fc', num=8))
    self.layers.append(Softmax(num=9))
    self.num_layers = len(self.layers)    
    return self.layers, self.num_layers

