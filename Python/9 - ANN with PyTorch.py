#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 22:38:02 2020

@author: batuhan
"""

import torch
import torchvision
from torchvision import transforms, datasets

train = datasets.MNIST("",train=True,download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("",train=False,download=True,
                      transform=transforms.Compose([transforms.ToTensor()]))
#%%

train_set = torch.utils.data.DataLoader(train,batch_size=3,shuffle=True)
test_set  = torch.utils.data.DataLoader(test,batch_size=3,shuffle=True)

import torch.nn as nn
import torch.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(28*28,8)
        self.net2 = nn.Linear(8,8)
        self.net3 = nn.Linear(8,8)
        self.net4 = nn.Linear(8,10)
        
    def forward(self,x):
        x=nn.relu(self.net1(x))
        x=nn.relu(self.net2(x))
        x=nn.relu(self.net3(x))
        x=self.net4(x)
        return F.log_softmax(x,dim=1)
        
        
        
        
net = Net()
print(net)
