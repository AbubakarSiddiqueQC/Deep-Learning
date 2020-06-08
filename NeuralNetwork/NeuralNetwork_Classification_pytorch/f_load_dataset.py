import torchvision
import torch
from torch.utils import data
from torchvision import transforms


def load_dataset(batch_size):   
    ### To generate figure 5 and 6 use this dataset
    
    # MNIST dataset 
    train_dataset = torchvision.datasets.MNIST(root='/data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

    test_dataset = torchvision.datasets.MNIST(root='/data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

    print("train_dataset", train_loader)
    print("test_dataset", test_loader)
    
    
    
    return train_loader, test_loader