import torch.nn as nn
import torch
import numpy as np
from model import Policy_Network, Value_Network
from config import *

class Agent:
    def __init__(self, gamma, lr_policy,lr_value, input_dims, output_dims,include_baseline=False):
        self.gamma = gamma
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.include_baseline = include_baseline
        self.policy = Policy_Network(lr_policy, input_dims, output_dims)
        self.policy = self.weight_init(self.policy)        

        if self.include_baseline:
            self.baseline_network = Value_Network(lr_value, input_dims, output_dims)
            self.baseline_network = self.weight_init(self.baseline_network)

    def weight_init(self,model):
        for p in model.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
        return model


    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def discounted_returns(self, rewards):
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
        return returns

    def learn(self, batch_trajectory):
        policy_losses = []
        value_losses = []

        for log_probs, rewards, states, next_states, dones in batch_trajectory:
            G = self.discounted_returns(rewards)
            log_probs = torch.stack(log_probs)

            if not self.include_baseline:
                
                policy_losses.append((-log_probs * G).sum())

            elif self.include_baseline:
                B = []
                for t in range(len(rewards)):
                    state_t = torch.tensor(states[t], dtype=torch.float32).to(DEVICE)
                    next_state_t = torch.tensor(next_states[t], dtype=torch.float32).to(DEVICE)
                    reward_t = torch.tensor(rewards[t], dtype=torch.float32).to(DEVICE)

                    done = dones[t]
                    curr_value = self.baseline_network(state_t).squeeze(0)

                    with torch.no_grad():                        
                        next_value = torch.tensor(0.0, device=DEVICE) if done else self.baseline_network(next_state_t).squeeze(0)

                    # TD(0) Target
                    td_target = reward_t + self.gamma * next_value
                    td_loss = self.baseline_network.loss(curr_value, td_target)
                    value_losses.append(td_loss)

                    # Policy gradient with MC return
                    b = G[t] - curr_value.detach()                    
                    policy_loss = -log_probs[t] * b
                    policy_losses.append(policy_loss)

        # Update policy network
        self.policy.optimizer.zero_grad()
        total_policy_loss = torch.stack(policy_losses).sum()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=GRAD_MAX_NORM)
        self.policy.optimizer.step()

        # Update value network if using baseline
        if self.include_baseline:
            total_value_loss = torch.stack(value_losses).mean()
            total_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.baseline_network.parameters(), max_norm=GRAD_MAX_NORM)
            self.baseline_network.optimizer.step()