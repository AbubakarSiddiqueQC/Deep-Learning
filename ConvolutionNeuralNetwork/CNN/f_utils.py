import numpy as np
import copy

def cross_entropy(outputs, targets):
    #return np.divide(np.mean(np.square(np.subtract(targets, outputs))), 2)
    N = outputs.shape[1]
    targets = targets.reshape(10,1)
    ce = -np.sum(targets * np.log(outputs)) / N
    return ce
    
