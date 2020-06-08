import numpy as np


def tanh(a):
    t=(np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a))
    return t
    #return  np.tanh(a)

def tanh_derivative(a):
    t=(np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a))
    dt=1-t**2
    return dt
    #return 1 - np.tanh(a)**2

def relu(x):
    return x * (x > 0)
    #if(a > 0):
    #    return a
    #else:
    #    return 0
  
def relu_derivative(x): 
    #if(np.maximum(0,a) == a):
    #    return 1
    #else:
    #    return 0
    return 1. * (x > 0)
def lrelu(a, k=0.1):
    output = np.copy( a )
    output[ output < 0 ] *= k
    return output

def lrelu_derivative(a, k = 0.1):  
    return np.clip(a > 0, k, 1.0)
def sigmoid(a):
    return 1/(1 + np.exp(-a))

def sigmoid_derivative(a):
    return np.exp(-a) / ((1 + np.exp(-a)) ** 2)

def softmax(x): 
#Compute softmax values for each sets of scores in x.

    e_x = np.exp(x - np.max(x)) 

    return e_x / e_x.sum(axis=0) 





