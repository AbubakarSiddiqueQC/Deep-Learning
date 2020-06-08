import numpy as np
import matplotlib.pyplot as plt
from neural_network import *
from f_load_dataset import load_dataset
def activation_functions_vs_errors():
    # load dataset
    train_x, train_t, val_x, val_t, test_x, test_t = load_dataset()
    
    # create l-dim network by just adding num of neurons in layer_dim
    # first and last elements represent input and output layers dim
    layer_dim = [1,50,50,50,50,50,1]
    
    # add activation functions name here. 
    # input layer activation function is None
    activations = [[None, 'sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','identity'],
                   [None, 'tanh','tanh','tanh','tanh','tanh','identity'],
                   [None, 'relu','relu','relu','relu','relu','identity'],
                   [None, 'lrelu','lrelu','lrelu','lrelu','lrelu','identity']]
    assert len(layer_dim) ==  len(activations[0]), "layer dim or activation is missing.."
    
    # hyper parameters of neural network
    learning_rate = 1e-3
    num_epochs = 100
    mini_batch_size = 25
    train_loss = []
    test_loss = []
    for l in range(4):
        nn = NeuralNetwork(layer_dim, activations[l], learning_rate, num_epochs, mini_batch_size)
    
    # train neural network 
        nn.train(train_x, train_t, val_x, val_t)
        # test neural network 
        trainloss, _ = nn.test(train_x, train_t)
        train_loss.append(trainloss)
        testloss, test_output = nn.test(test_x, test_t)
        test_loss.append(testloss)
    
    
    ########################################################################################
    # learning_rate = 1e-3
    # epochs = 100
    # hidden layers = 5
    # num of neurons = 100
    # mini_batch_size = 25
    # activation functions vs errors
    #train_loss = []
    #test_loss = []
activation_functions_vs_errors()
width = 0.3
act_func = ['sigmoid', 'tanh', 'relu', 'lrelu']
fig = plt.gcf()
plt.xticks(range(len(act_func)), act_func)
plt.xlabel('activation functoins')
plt.ylabel('loss')
plt.bar(np.arange(len(train_loss)),train_loss, width=width, label='train loss') 
plt.bar(np.arange(len(test_loss))+width,test_loss, width=width, label='test loss') 
plt.legend()
plt.show()
fig.savefig('Figure_4', dpi=300)


########################################################################################
test_error_sigmoid = []
test_error_tanh = []
test_error_relu = []
test_error_lrelu = []
def activation_functions_vs_hidden_layers():
    # load dataset
    train_x, train_t, val_x, val_t, test_x, test_t = load_dataset()
    
    # create l-dim network by just adding num of neurons in layer_dim
    # first and last elements represent input and output layers dim
    layer_dim = [[1,50,1],[1,50,50,1],[1,50,50,50,1],[1,50,50,50,50,50,1],[1,50,50,50,50,50,50,50,50,50,50,1]]
    
    # add activation functions name here. 
    # input layer activation function is None
    activations = [[[None, 'sigmoid','identity'],
                   [None, 'tanh','identity'],
                   [None, 'relu','identity'],
                   [None, 'lrelu','identity']],
                   [[None, 'sigmoid','sigmoid','identity'],
                   [None, 'tanh','tanh','identity'],
                   [None, 'relu','relu','identity'],
                   [None, 'lrelu','lrelu','identity']],
                   [[None, 'sigmoid','sigmoid','sigmoid','identity'],
                   [None, 'tanh','tanh','tanh','identity'],
                   [None, 'relu','relu','relu','identity'],
                   [None, 'lrelu','lrelu','lrelu','identity']],
                   [[None, 'sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','identity'],
                   [None, 'tanh','tanh','tanh','tanh','tanh','identity'],
                   [None, 'relu','relu','relu','relu','relu','identity'],
                   [None, 'lrelu','lrelu','lrelu','lrelu','lrelu','identity']],
                   [[None, 'sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','identity'],
                   [None, 'tanh','tanh','tanh','tanh','tanh','tanh','tanh','tanh','tanh','tanh','identity'],
                   [None, 'relu','relu','relu','relu','relu','relu','relu','relu','relu','relu','identity'],
                   [None, 'lrelu','lrelu','lrelu','lrelu','lrelu','lrelu','lrelu','lrelu','lrelu','lrelu','identity']]]
    
    #hyper parameters of neural network
    learning_rate = 1e-3
    epochs = 100
    mini_batch_size = 10
    for i in range(5):
        for j in range(4):
            nn = NeuralNetwork(layer_dim[i], activations[i][j], learning_rate, num_epochs, mini_batch_size)
            # train neural network 
            nn.train(train_x, train_t, val_x, val_t)
            trainloss, _ = nn.test(train_x, train_t)
            train_loss.append(trainloss)
            testloss, test_output = nn.test(test_x, test_t)
            # test neural network 
            if(j == 0):
                test_error_sigmoid.append(testloss)
            elif(j == 1):
                test_error_tanh.append(testloss)
            elif(j == 2):
                test_error_relu.append(testloss)
            else:
                test_error_lrelu.append(testloss)
activation_functions_vs_hidden_layers()
#activation functions vs hidden layers
act_func = ['1HL', '2HL', '3HL', '5HL', '10HL']
fig = plt.gcf()
plt.xticks(range(len(act_func)), act_func)
plt.xlabel('hidden layers')
plt.ylabel('loss')
plt.plot(test_error_sigmoid, color='darkorange',linestyle='dashed',linewidth=2, marker='o', markerfacecolor='darkorange', markersize=8, label='sigmoid') 
plt.plot(test_error_tanh, color='red',linestyle='dashed',linewidth =2, marker='o', markerfacecolor='red', markersize=8, label='tanh') 
plt.plot(test_error_relu, color='purple',linestyle='dashed',linewidth =2, marker='o', markerfacecolor='purple', markersize=8, label='relu') 
plt.plot(test_error_lrelu, color='blue',linestyle='dashed',linewidth =2, marker='o', markerfacecolor='blue', markersize=8, label='lrelu') 
plt.legend()
plt.show()
fig.savefig('Figure_5', dpi=300)

########################################################################################








