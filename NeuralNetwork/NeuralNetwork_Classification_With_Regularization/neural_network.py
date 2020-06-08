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
        self.lastgrads = dict()
        self.adammom = dict()
        self.adamv = dict()
        self.vals_for_backp = dict()
        self.lembda = 0.0
        self.drop = dict()
        

        
    def initialize_parameters(self):
        print("Num of layers", self.num_layers)
        for l in range(1, self.num_layers + 1):
            if self.activations_func[l] == 'relu': #xavier intialization method
                self.parameters['W%s' % l] = np.random.randn(self.num_neurons[l], self.num_neurons[l-1]) / np.sqrt(self.num_neurons[l-1]/2.)
            else:                
                self.parameters['W%s' % l] = np.random.randn(self.num_neurons[l], self.num_neurons[l-1]) / np.sqrt(self.num_neurons[l - 1])
            self.parameters['b%s' % l] = np.zeros((self.num_neurons[l], 1))
            self.lastgrads["dW%s"%l] = 0
            self.lastgrads["db%s"%l] = 0
            self.adammom["dW%s"%l] = 0
            self.adammom["db%s"%l] = 0
            self.adamv["dW%s"%l] = 0
            self.adamv["db%s"%l] = 0
            print("weights and biases initialization",self.parameters['W%s'%l].shape,self.parameters['b%s'%l].shape)
          
    
    def fprop(self, batch_input,traing = False,keep_prob = None):
        self.net1['A0'] = batch_input
        for i in range(1, self.num_layers + 1):
            self.net['Z%s' %i] = self.parameters['W%s' %i].dot(self.net1['A%s'%(i-1)]) + self.parameters['b%s' %i]
            self.net1['A%s' %i] = eval(self.activations_func[i])(self.net['Z%s'%i])
            #Dropout
            if(traing):
                self.drop[str(i)] = np.random.rand(self.net1['A%s' %i].shape[0], self.net1['A%s' %i].shape[1])
                self.drop[str(i)] = self.drop[str(i)] < keep_prob[i]
                self.net1['A%s' %i] = np.multiply(self.net1['A%s' %i], self.drop[str(i)])
                self.net1['A%s' %i] /= keep_prob[i]
            
             
    
             
            
            
    def calculate_loss(self, batch_target,regu = False):
        #return np.divide(np.mean(np.square(np.subtract(batch_target, self.net1['A%s' %self.num_layers]))), 2)
        N = self.net1['A%s' %self.num_layers].shape[1]
        ce = -np.sum(batch_target * np.log(self.net1['A%s' %self.num_layers])) / N
        #l2 regulization
        if(regu):
            for i in range(1,self.num_layers+1):
                ce += (self.lembda/2) * np.sum(np.square(self.parameters["W%s" % i]))
            
                
        return ce
        
    def update_parameters_adam(self,epoch):
        beta1 = 0.9
        beta2 = 0.999
        eps_stable = 1e-8
        
        for i in range(1,self.num_layers+1):
            
            self.adammom["dW%s"%i] = (beta1 * self.adammom["dW%s"%i]) + ((1. - beta1) * self.grads["dW%s"%i])
            self.adammom['db%s'%i] = (beta1 * self.adammom['db%s'%i]) + ((1. - beta1) * self.grads["db%s"%i])
            
            self.adamv["dW%s"%i] = (beta2 * self.adamv["dW%s"%i]) + ((1. - beta2) * np.square(self.grads["dW%s"%i]))
            self.adamv['db%s'%i] = (beta2 * self.adamv['db%s'%i]) + ((1. - beta2) * np.square(self.grads["db%s"%i]))
            
            #sqr[:] = beta2 * sqr + (1. - beta2) * np.square(g)

            mom_prime_w = self.adammom["dW%s"%i] / (1. - (beta1 ** i))
            mom_prime_b = self.adammom['db%s'%i] / (1. - (beta1 ** i))
            vm = self.adamv["dW%s"%i] / (1. - (beta2 ** i)) 
            vb = self.adamv['db%s'%i] / (1. - (beta2 ** i)) 

            update_w = (self.learning_rate / (np.sqrt(vm) + eps_stable)) * mom_prime_w
            update_b = (self.learning_rate / (np.sqrt(vb) + eps_stable)) * mom_prime_b
            #param[:] = param - div
            self.parameters['W%s' %i] = self.parameters['W%s' % i] - update_w
            self.parameters['b%s' %i] = self.parameters['b%s' % i] - update_b
            
    def update_parameters_momentum(self,epoch):
        #will store last self.grads in self.lastgrads in bprob
        
        for i in range(1,self.num_layers+1):
            self.parameters['W%s' %i] = self.parameters['W%s' % i] - (self.learning_rate * self.lastgrads["dW%s"%i])
            self.parameters['b%s' %i] = self.parameters['b%s' % i] - (self.learning_rate * self.lastgrads['db%s'%i])
        
    
    def update_parameters(self,epoch):
        for i in range(1,self.num_layers+1):
            self.parameters["W%s" %i] = self.parameters["W%s" % i] - (self.learning_rate * self.grads["dW%s"%i])
            self.parameters['b%s' %i] = self.parameters['b%s' %i ] - (self.learning_rate * self.grads['db%s'%i])
    
    def bprop(self, batch_target,epoch,traing = False,keep_prob = None,regu = False,):
        beta = 0.9
        output_error = self.net1['A%s' %self.num_layers] - batch_target
        
        self.grads['dW%s' %self.num_layers] = output_error.dot(self.net1['A%s' %(self.num_layers-1)].T)
        self.grads['db%s' %self.num_layers] = np.sum(output_error, axis = 1, keepdims = True)
        #l2 regulization
        if(regu):
                self.grads['dW%s'%self.num_layers] += self.lembda * self.parameters["W%s" %self.num_layers] 
        #for momentum
        self.lastgrads["dW%s"%self.num_layers] = (beta * self.lastgrads["dW%s"%self.num_layers]) + self.grads['dW%s' %self.num_layers]
        self.lastgrads["db%s"%self.num_layers] = (beta * self.lastgrads["db%s"%self.num_layers]) + self.grads['db%s' %self.num_layers]
        last_hidden_error = self.parameters['W%s'%self.num_layers] .T.dot(output_error)
        for i in reversed(range(1, self.num_layers)):
            
            dz = np.multiply(last_hidden_error,eval(self.activations_func[1]+'_derivative')(self.net['Z%s'%i]))
            #Dropout
            if(traing):
                dz = np.multiply(dz, self.drop[str(i)])
                dz /= keep_prob[i]
            self.grads['dW%s' %i] = dz.dot(self.net1['A%s' %(i-1)].T)
            self.grads['db%s' %i] = np.sum(dz, axis = 1, keepdims = True)
            #l2 regulization
            if(regu):
                self.grads['dW%s' %i] += self.lembda * self.parameters["W%s" %i] 
            #for momentum
            self.lastgrads["dW%s"%i] = (beta * self.lastgrads["dW%s"%i]) + self.grads['dW%s' %i]
            self.lastgrads["db%s"%i] = (beta * self.lastgrads["db%s"%i]) + self.grads['db%s' %i]
            
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
    

    def train(self, train_x, train_y,keep_prob):
        
        train_x, train_y = shuffle(train_x, train_y, random_state=0)
        self.initialize_parameters()        
        train_loss = []
        val_loss = []  
        num_samples = train_y.shape[0]
        
        
        

        for i in range(0, self.num_iterations):
            for idx in range(0, num_samples, self.mini_batch_size):
                minibatch_input =  train_x[idx:idx + self.mini_batch_size,:]
                minibatch_target =  train_y[idx:idx +self.mini_batch_size,:]
                self.fprop(minibatch_input.T,True,keep_prob)
                loss = self.calculate_loss(minibatch_target.T)
                self.bprop(minibatch_target.T,i,True,keep_prob)           
                self.update_parameters(i)
                acc,count = self.test(minibatch_input,minibatch_target)
                   
            train_loss.append(loss) 
            #self.fprop(val_x)
            #va_loss = self.calculate_loss(val_y)
            #val_loss.append(va_loss) 
            print("Epoch %i: training loss %f Training Accuracy %f" % (i, loss,acc))
        self.plot_loss(train_loss,val_loss)      
        self.plot_gradients()
        
    def test(self,x,y):
        correct_count, all_count = 0, 0
        pred_labels = []
        true_labels = [] 
        for n in range(len(x)):
            self.fprop(np.reshape(x[n], (784,1)))
            pred_label = self.net1['A%s'%self.num_layers].argmax()
            true_label = y[n].argmax()
            pred_labels.append(pred_label)
            true_labels.append(true_label)
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1
        return (100*(correct_count/all_count)),all_count
        
         
         

   