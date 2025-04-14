from agent import Agent
from config import *
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt

def moving_average(rewards, window=100):
    return np.convolve(rewards, np.ones(window)/window, mode='valid')

def train_dueling_dqn(env_name,seed,epsiodes,steps,save_model=False,ddqn_type=None):
    env = gym.make(env_name)
    if seed is not None:
        env.reset(seed=seed)

    n_actions = env.action_space.n
    input_dims = env.observation_space.shape[0]

    agent = Agent(gamma=GAMMA,epsilon=EPSILON,input_dims=input_dims,batch_size=BATCH_SIZE,lr=ALPHA,
                  n_actions=n_actions,eps_end=EPSILON_END,max_mem_size=MEM_SIZE,target_update_freq=TARGET_NET_FREQ,ddqn_type=ddqn_type)
    n_episodes = epsiodes
    train_scores = []
    test_scores = []

    for episode in range(n_episodes):
        observation, _ = env.reset()
        done = False
        train_score = 0

        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.store_transitions(
                state=torch.tensor(observation, dtype=torch.float),
                reward=torch.tensor(reward, dtype=torch.float),
                action=torch.tensor(action, dtype=torch.long),
                next_state=torch.tensor(next_observation, dtype=torch.float),
                done=torch.tensor(done, dtype=torch.bool)
            )

            loss = agent.learn()
            observation = next_observation
            train_score += reward
        train_scores.append(train_score)
        train_avg_score = np.mean(train_scores[-100:])

        observation,_ = env.reset()
        done = False
        test_score = 0
        old_eps = agent.epsilon
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated            
            observation = next_observation
            test_score += reward
            agent.epsilon = 0.05
        
        
        test_scores.append(test_score)
        test_avg_score = np.mean(test_scores[-100:])
        agent.epsilon = old_eps


        print(f'\rEpisode {episode + 1}/{n_episodes} | '
            f'Training-Score: {train_score:.1f} | Train Avg Score (last 100): {train_avg_score:.2f} | '
            f'Epsilon: {agent.epsilon:.3f} | ',f'Testing-score: {test_score:.3f} | ',f'Test Avg score (last 100): {test_avg_score:.3f} | ', end='', flush=True)

    print()
    if save_model:
        model_path = './models/dueling_dqn_{}_acrobot.pt'.format(ddqn_type)
        torch.save({
            'model_state_dict':agent.Q_network.state_dict(),
            'optimizer_state_dict':agent.Q_network.optimizer.state_dict(),
            'epsilon':agent.epsilon
        },model_path)
        print("Model saved to {} ".format(model_path))
    return train_scores,test_scores

if __name__ == '__main__':
    train_mean_score, test_mean_score = train_dueling_dqn('Acrobot-v1',seed=None,epsiodes=200,steps=100,save_model=True,ddqn_type='mean')
    train_max_score, test_max_score = train_dueling_dqn('Acrobot-v1',seed=None,epsiodes=200,steps=100,save_model=True,ddqn_type='max')