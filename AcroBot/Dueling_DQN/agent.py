import torch
import numpy as np
from model import Dueling_DQN
from utils import ReplayBuffer

class Agent():
    def __init__(self,gamma,epsilon,lr,input_dims,batch_size,n_actions,eps_end=0.01,max_mem_size=100000,target_update_freq=100,ddqn_type='mean'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.lr = lr
        self.n_actions = n_actions
        self.eps_end = eps_end
        self.mem_size = max_mem_size
        self.replay_buffer = ReplayBuffer(self.mem_size,self.input_dims)
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.target_update_freq = target_update_freq
        self.Q_network = Dueling_DQN(self.lr,self.input_dims,256,256,self.n_actions,type=ddqn_type)
        self.Target_network = Dueling_DQN(self.lr,self.input_dims,256,256,self.n_actions,type=ddqn_type)
        self.Target_network.load_state_dict(self.Q_network.state_dict())

    def choose_action(self,observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.Q_network.device)
            n_actions = self.Q_network.forward(state)
            action = torch.argmax(n_actions).item()
        return action
    
    def replace_target_network(self):
        self.Target_network.load_state_dict(self.Q_network.state_dict())

    def learn(self):
        if self.replay_buffer.mem_cntr < self.batch_size: return

        self.Q_network.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_buffer(self.batch_size)
        device = self.Q_network.device

        states = states.to(device)
        actions = actions.long().to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        indices = torch.arange(self.batch_size).to(device)
        Q_pred = self.Q_network(states)[indices, actions]

        next_actions = self.Q_network(next_states).argmax(dim=1)

        Q_next = self.Target_network(next_states)[indices, next_actions]
        Q_next[dones] = 0.0


        Q_target = rewards + self.gamma * Q_next

        loss = self.Q_network.loss(Q_pred, Q_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q_network.parameters(), max_norm=10)
        self.Q_network.optimizer.step()

        self.epsilon = max(self.eps_end, self.epsilon*0.995)
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.replace_target_network()
        return loss.item()


