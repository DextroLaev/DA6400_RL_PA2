import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import *

class Policy_Network(nn.Module):
    def __init__(self, lr, input_dims, output_dims):
        super().__init__()
        self.fc1 = nn.Linear(input_dims,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,output_dims)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.to(DEVICE)

    def forward(self, state):
        x = (F.relu(self.fc1(state)))
        x = (F.relu(self.fc2(x)))
        x = (self.fc3(x))      
        return F.softmax(x, dim=-1)


class Value_Network(nn.Module):
    def __init__(self,lr,input_dims,output_dims):
        super(Value_Network,self).__init__()
        self.fc1 = nn.Linear(input_dims,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,1)                
        self.dropout =nn.Dropout(0.2)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.to(DEVICE)
    
    def forward(self,state):
        x = (F.relu(self.fc1(state)))
        x = (F.relu(self.fc2(x)))
        x = (F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x