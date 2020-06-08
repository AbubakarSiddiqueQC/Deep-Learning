import numpy as np
from f_utils import *
import copy
def fpropforchecking(self,batch_input,batch_target,parameters1):
    self.net1['A0'] = batch_input
    for i in range(1, self.num_layers + 1):
        self.net['Z%s' %i] = parameters1['W%s' %i].dot(self.net1['A%s'%(i-1)]) + parameters1['b%s' %i]
        if self.activations_func[i]=="identity":# in this case output layer which activation is just output
            # Output layer, no activation
            self.net1['A%s' %i] = self.net['Z%s' %i]
        else:
            # Hidden layers, activations_func[i] activataion
            self.net1['A%s' %i] = eval(self.activations_func[i])(self.net['Z%s'%i])
    return np.divide(np.square(np.subtract(batch_target, self.net1['A%s' %self.num_layers])), 2)  


def check_gradients(self, train_X, train_t):
    # Roll out parameters and gradients dictionaries
    self.fprop(train_X)
    self.bprop(train_t)             
    eps= 1e-5
    grad_ok = 0
    theta_plus = copy.deepcopy(self.parameters)
    theta_minus = copy.deepcopy(self.parameters)
    for l in range(1, self.num_layers+1):
        theta_plus['W%s'%l] = theta_plus['W%s'%l] + eps
        theta_plus['b%s'%l] = theta_plus['b%s'%l] + eps
        j_plus = fpropforchecking(self, train_X,train_t,theta_plus)
        theta_minus['W%s'%l] = theta_minus['W%s'%l] - eps
        theta_minus['b%s'%l] = theta_minus['b%s'%l] - eps
        j_minus = fpropforchecking( self,train_X,train_t,theta_minus)
        Numerical_grad = (j_plus - j_minus) / (2 * eps)
        Analytical_grad = self.grads["dW%s"%l]
        diff = (np.linalg.norm(Numerical_grad - Analytical_grad))  / (np.linalg.norm(Numerical_grad) + np.linalg.norm(Analytical_grad))
        
                              
        if (diff> eps):
            print("layer %s gradients are not ok"% l)  
            grad_ok = 0
        else:
            print("layer %s gradients are ok"% l)
            grad_ok = 1
              
    return grad_ok
         
            
        
        
        
        