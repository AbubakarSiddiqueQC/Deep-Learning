import numpy as np
from sklearn.metrics import confusion_matrix
from layers import *
from f_model import define_model
from  f_utils import *

class Network:
    def __init__(self, batch_size, val_batch_size, epoch):
        self.layers = []
        self.layer = Layers()
        self.layers, self.num_layers = define_model(self, self.layers)         
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.epochs = epoch
        self.training_acc = 0
        self.val_acc = 0
        self.loss = 0
        self.valloss = 0


    def cal_val_acc(self, val_inputs, val_targets, e):
         for idx in range(0, val_inputs.shape[0], self.val_batch_size):
             val_input =  val_inputs[idx:idx + self.val_batch_size]
             val_target =  val_targets[idx:idx + self.val_batch_size]
             self.total_acc = 0 
             count = 0
                  
             for i in range(self.val_batch_size):
                val_x = val_input[i]
                val_t = val_target[i]
                
                for l in range(self.num_layers):
                    output = self.layers[l].forward(val_x, l+1)
                    val_x = output
                pred_label = output.argmax()
                true_label = val_t.argmax()
                if(true_label == pred_label):
                    self.total_acc += 1
                count += 1
                self.valloss += cross_entropy(output, val_t)
                
             self.val_accuracy = (100*(self.total_acc/count))
             self.valloss = self.valloss /count
             return self.val_accuracy
              
    
    def train(self, train_inputs, train_targets, val_inputs, val_targets):
        for e in range(self.epochs):
            for idx in range(0,  train_inputs.shape[0], self.batch_size):
                batch_input =  train_inputs[idx:idx + self.batch_size]
                batch_target =  train_targets[idx:idx +self.batch_size]
                self.total_acc = 0 
                count = 0
                self.loss = 0 
                no_of_img = batch_input.shape[0]
                for i in range(no_of_img):
                    train_x = batch_input[i]
                    train_t = batch_target[i]                 
                   
                    
                    for l in range(self.num_layers):                                                        
                        output = self.layers[l].forward(train_x, l+1)
                        train_x = output
                    
                    self.loss += cross_entropy(output, train_t)
                    
                    pred_label = output.argmax()
                    true_label = train_t.argmax()
            
                    if(true_label == pred_label):
                        self.total_acc += 1
                    count += 1

        
                    

                        
                    dy = train_t                                   
                    for l in range(self.num_layers-1, -1, -1):    
                        dz = self.layers[l].backward(dy, l+1) 
                        dy = dz
                                            
                    for l in range(self.num_layers):                                                        
                        if(self.layers[l].name == "conv" or self.layers[l].name == "fc"):
                            self.layers[l].update_parameters(l+1)
                self.loss = self.loss / no_of_img
                self.training_acc = (100*(self.total_acc/count))
                self.val_acc = self.cal_val_acc(val_inputs, val_targets, e)
                print('Epoch: {0:d}/{1:d} --- Iter:{2:d} -- Loss: {3:.2f} --- training accuracy: {4:.2f} %  --- val accuracy: {5:.2f} %'.format(e, self.epochs, idx+self.batch_size, self.loss, self.training_acc, self.val_acc))
   
                     
    def test(self, test_inputs, test_targets):       
        self.total_acc = 0 
        count = 0
        pred_labels = []
        true_labels = [] 
        for i in range(test_inputs.shape[0]):
            test_x = test_inputs[i]
            test_t = test_targets[i]
            for l in range(self.num_layers):
                output = self.layers[l].forward(test_x, l+1)   
                test_x = output
            pred_label = output.argmax()
            true_label = test_t.argmax()
            pred_labels.append(pred_label)
            true_labels.append(true_label)
            if(true_label == pred_label):
                self.total_acc += 1
            count += 1

        
        acc = (100*(self.total_acc/count))
        print('Testing accuracy:', acc)
        confusion_mat = confusion_matrix(true_labels, pred_labels)
        print("Confusion Matrx =",confusion_mat )
        print("Accuracy of each class")
        print(100*confusion_mat.diagonal()/confusion_mat.sum(1))    


  
