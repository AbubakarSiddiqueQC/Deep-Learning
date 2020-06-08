import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torchsummary import summary
from time import time
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_loss(loss,val_loss, len_layer, learning_rate):        
    plt.figure()
    fig = plt.gcf()
    plt.plot(loss, linewidth=3, label="train")
    plt.plot(val_loss, linewidth=3, label="val")
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('learning rate =%s, hidden layers=%s' % (learning_rate, len_layer-1))
    plt.grid()
    plt.legend()
    plt.show()
    fig.savefig('plot_loss.png')
        
    
def plot_gradients(net, len_layer):
    avg_l_g = []
    for idx, param in enumerate(net.parameters()):
        if idx % 2 == 0:
             weights_grad = param.grad 
             dim = weights_grad.shape[0]
             avg_g = []
             for d in range(dim):
                 abs_g = np.abs(weights_grad[d].numpy())           
                 avg_g.append(np.mean(abs_g))             
             temp = np.mean(avg_g)
             avg_l_g.append(temp)   
    layers = ['layer %s'%l for l in range(len_layer+1)]
    weights_grad_mag = avg_l_g
    fig = plt.gcf()
    plt.xticks(range(len(layers)), layers)
    plt.xlabel('layers')
    plt.ylabel('average gradients magnitude')
    plt.title('')
    plt.bar(range(len(weights_grad_mag)),weights_grad_mag, color='red', width=0.2) 
    plt.show() 
    fig.savefig('plot_gradients.png')
    

class NeuralNet(nn.Module):
    def __init__(self, size_list, activations):
        super(NeuralNet, self).__init__()
        layers = []
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            if activations[i+1] == 'sigmoid':
                act = nn.Sigmoid()
            elif activations[i+1] =='tanh':
                act = nn.Tanh()
            elif activations[i+1] == 'relu':
                act = nn.ReLU()
            layers.append(act)
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        #because this is classification
        layers.append(nn.LogSoftmax(dim=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x

def train(train_loader, learning_rate, num_epochs, layer_dim, activations):
    train_loss = []
    val_loss = []  
    time0 = time()

    len_layer = len(layer_dim) - 1
    
    net = NeuralNet(layer_dim, activations)    
    calculate_loss = nn.NLLLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    print(net)

    
 
    
    total_parameters = sum(p.numel() for p in net.parameters())
    print("total number of parameters:", total_parameters)
    #summary(net, (784//2,128))
    
    
    for i in range(0, num_epochs):
        running_loss = 0
        for images, labels in train_loader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

            # Forward pass
            outputs = net(images)
            loss = calculate_loss(outputs,labels)
            
            # Backprop and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()                
            running_loss += loss.item()
        train_loss.append(loss)       
        print("Epoch {} - Training loss: {}".format(i, running_loss/len(train_loader)))
    
    print("\nTraining Time (in minutes) =",(time()-time0)/60)    
    plot_loss(train_loss,val_loss, len_layer, learning_rate)      
    plot_gradients(net, len_layer)
    
    checkpoint = {'model': NeuralNet(layer_dim, activations),
                  'state_dict': net.state_dict(),
                  'optimizer' : optimizer.state_dict()}
    torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model
    
def test(test_loader):
    pred_labels = []
    true_labels = [] 
    net = load_checkpoint('checkpoint.pth')
    correct_count, all_count = 0, 0
    for images,labels in test_loader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                logps = net(img)
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            pred_labels.append(pred_label)
            true_labels.append(true_label)
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", 100*(correct_count/all_count))
    confusion_mat = confusion_matrix(true_labels, pred_labels)
    print("Confusion Matrx =",confusion_mat )
    print("Accuracy of each class")
    print(100*confusion_mat.diagonal()/confusion_mat.sum(1))    