import numpy as np
import matplotlib.pyplot as plt
from f_utils import *
import copy
from f_check_gradient import *
from sklearn.utils import shuffle


class NeuralNetwork():
  
    def __init__(self, num_neurons, activations_func, learning_rate, num_epochs, mini_batch_size):     
        self.num_neurons = num_neurons
        self.activations_func = activations_func
        self.learning_rate = learning_rate
        self.num_iterations = num_epochs
        self.mini_batch_size = mini_batch_size
        self.num_layers = len(self.num_neurons) - 1
        self.parameters = dict()
        self.net = dict()
        self.net1 = dict()
        self.grads = dict()
        self.vals_for_backp = dict()

        
    def initialize_parameters(self):
        print("Num of layers", self.num_layers)
        for l in range(1, self.num_layers + 1):
            if self.activations_func[l] == 'relu': #xavier intialization method
                self.parameters['W%s' % l] = np.random.randn(self.num_neurons[l], self.num_neurons[l-1]) / np.sqrt(self.num_neurons[l-1]/2.)
            else:                
                self.parameters['W%s' % l] = np.random.randn(self.num_neurons[l], self.num_neurons[l-1]) / np.sqrt(self.num_neurons[l - 1])
            self.parameters['b%s' % l] = np.zeros((self.num_neurons[l], 1))
            print("weights and biases initialization",self.parameters['W%s'%l].shape,self.parameters['b%s'%l].shape)

          
    def fprop(self, batch_input):
        self.net1['A0'] = batch_input
        for i in range(1, self.num_layers + 1):
            self.net['Z%s' %i] = self.parameters['W%s' %i].dot(self.net1['A%s'%(i-1)]) + self.parameters['b%s' %i]
            if self.activations_func[i]=="identity":# in this case output layer which activation is just output
                # Output layer, no activation
                self.net1['A%s' %i] = self.net['Z%s' %i]
            else:
                # Hidden layers, activations_func[i] activataion
                self.net1['A%s' %i] = eval(self.activations_func[i])(self.net['Z%s'%i])
             
    def calculate_loss(self, batch_target):
        return np.divide(np.mean(np.square(np.subtract(batch_target, self.net1['A%s' %self.num_layers]))), 2)
        
        
    def update_parameters(self,epoch):
        for i in range(1,self.num_layers+1):
            self.parameters["W%s" %i]=self.parameters["W%s" % i]-(self.learning_rate*self.grads["dW%s"%i])
            self.parameters['b%s' %i] = self.parameters['b%s'%i] - (self.learning_rate * self.grads['db%s'%i])
    
    def bprop(self, batch_target):
        output_error = self.net1['A%s' %self.num_layers] - batch_target
        self.grads['dW%s' %self.num_layers] = output_error.dot(self.net1['A%s' %(self.num_layers-1)].T)
        self.grads['db%s' %self.num_layers] = np.sum(output_error, axis = 1, keepdims = True)
        last_hidden_error = self.parameters['W%s'%self.num_layers] .T.dot(output_error)
        for i in reversed(range(1, self.num_layers)):
            dz = np.multiply(last_hidden_error,eval(self.activations_func[1]+'_derivative')(self.net['Z%s'%i]))
            self.grads['dW%s' %i] = dz.dot(self.net1['A%s' %(i-1)].T)
            self.grads['db%s' %i] = np.sum(dz, axis = 1, keepdims = True)
            last_hidden_error = self.parameters['W%s'%i] .T.dot(dz)

    def plot_loss(self,loss,val_loss):        
        plt.figure()
        fig = plt.gcf()
        plt.plot(loss, linewidth=3, label="train")
        plt.plot(val_loss, linewidth=3, label="val")
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.title('learning rate =%s, hidden layers=%s' % (self.learning_rate, self.num_layers-1))
        plt.grid()
        plt.legend()
        plt.show()
        fig.savefig('plot_loss.png')
        
    
    def plot_gradients(self):
        avg_l_g = []
        grad = copy.deepcopy(self.grads)
        for l in range(1, self.num_layers+1):
#             print("layer %s"%l)
             weights_grad = grad['dW%s' % l]  
             dim = weights_grad.shape[0]
             avg_g = []
             for d in range(dim):
                 abs_g = np.abs(weights_grad[d])
                 avg_g.append(np.mean(abs_g))             
             temp = np.mean(avg_g)
             avg_l_g.append(temp)   
        layers = ['layer %s'%l for l in range(self.num_layers+1)]
        weights_grad_mag = avg_l_g
        fig = plt.gcf()
        plt.xticks(range(len(layers)), layers)
        plt.xlabel('layers')
        plt.ylabel('average gradients magnitude')
        plt.title('')
        plt.bar(range(len(weights_grad_mag)),weights_grad_mag, color='red', width=0.2) 
        plt.show() 
        fig.savefig('plot_gradients.png')
    

    def train(self, train_x, train_y, val_x, val_y):
        train_x, train_y = shuffle(train_x, train_y, random_state=0)
        self.initialize_parameters()        
        train_loss = []
        val_loss = []  
        num_samples = train_y.shape[1]       
        check_grad = False
        grad_ok = 1
        

        for i in range(0, self.num_iterations):
            for idx in range(0, num_samples, self.mini_batch_size):
                minibatch_input =  train_x[:, idx:idx + self.mini_batch_size]
                minibatch_target =  train_y[:, idx:idx + self.mini_batch_size]
                
                if check_grad == True:
                    grad_ok = check_gradients(self, minibatch_input, minibatch_target)               
                    if grad_ok == 0:                           
                        print("gradients are not ok!\n")                           
                            
                   
                if grad_ok == 1:
                    check_grad = False
                    self.fprop(minibatch_input)
                    loss = self.calculate_loss(minibatch_target)
                    self.bprop(minibatch_target)           
                    self.update_parameters(i)
                   
            train_loss.append(loss) 
            self.fprop(val_x)
            va_loss = self.calculate_loss(val_y)
            val_loss.append(va_loss) 
            print("Epoch %i: training loss %f, validation loss %f" % (i, loss,va_loss))
        self.plot_loss(train_loss,val_loss)      
        self.plot_gradients()
        
    def test(self,x,y):
         self.fprop(x)
         loss=self.calculate_loss(y)
         # test_loss.append(loss)
         return loss,self.net1['A%s'%self.num_layers]

   