import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Dueling_DQN(nn.Module):
    def __init__(self,lr,input_dims,fc1_dims,fc2_dims,n_actions,type=None):
        super(Dueling_DQN,self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.ddqn_type = type
        self.fc1 = nn.Linear(self.input_dims,self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)

        #value function
        self.value_fc1 = nn.Linear(self.fc2_dims,64)        
        self.value_fn = nn.Linear(64,1)

        # advantage function
        self.adv_fc1 = nn.Linear(self.fc2_dims,128)
        self.adv_fc2 = nn.Linear(128,64)        
        self.advantage_fn = nn.Linear(64,self.n_actions)

        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self,state):
        x = self.fc1(state)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)

        # forward pass of value network
        val = F.gelu(self.value_fc1(x))
        val = self.value_fn(val)

        # forward pass of advantage network
        advantage = F.gelu(self.adv_fc1(x))   
        advantage = F.gelu(self.adv_fc2(advantage))     
        advantage = self.advantage_fn(advantage)


        if self.ddqn_type == 'mean':
            return val + (advantage - advantage.mean(dim=1,keepdim=True))
        elif self.ddqn_type == 'max':
            return val + (advantage - advantage.max(dim=1,keepdim=True).values)
