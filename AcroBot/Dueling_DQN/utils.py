import numpy as np
import torch

class ReplayBuffer:
    def __init__(self,max_size,input_dims):
        self.max_size = max_size
        self.mem_cntr = 0
        self.input_dims = input_dims
        self.state_buffer = torch.zeros((self.max_size,input_dims))
        self.next_state_buffer = torch.zeros((self.max_size,input_dims))
        self.action_buffer = torch.zeros(self.max_size)
        self.reward_buffer = torch.zeros(self.max_size)
        self.terminal_buffer = torch.zeros(self.max_size,dtype=torch.bool)
    
    def store_transitions(self,state,reward,action,next_state,done):
        index = self.mem_cntr % self.max_size
        self.state_buffer[index] = state
        self.next_state_buffer[index] = next_state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.terminal_buffer[index] = done
        self.mem_cntr += 1
    
    def sample_buffer(self,batch_size):
        max_mem = min(self.mem_cntr, self.max_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_buffer[batch]
        actions = self.action_buffer[batch]
        rewards = self.reward_buffer[batch]
        next_states = self.next_state_buffer[batch]
        dones = self.terminal_buffer[batch]
        return states, actions, rewards, next_states, dones
