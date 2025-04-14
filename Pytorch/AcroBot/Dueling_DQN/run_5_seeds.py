# run_plot.py
import numpy as np
import matplotlib.pyplot as plt
from train import train_dueling_dqn
from config import *

def moving_average(rewards, window=100):
    return np.convolve(rewards, np.ones(window)/window, mode='valid')

def run_multiple_seeds(env_name, episodes, steps, ddqn_type='mean', num_runs=5):
    all_runs_train = []
    all_runs_test = []
    for seed in range(num_runs):
        print(f'Run {seed+1} for ddqn_type={ddqn_type}')
        train_rewards,test_rewards = train_dueling_dqn(env_name, seed, episodes, steps, ddqn_type=ddqn_type)
        all_runs_train.append(moving_average(train_rewards, window=20))
        all_runs_test.append(moving_average(test_rewards, window=20))
    return np.array(all_runs_train),np.array(all_runs_test)


def plot_mean_std(all_rewards, label, color):
    mean = np.mean(all_rewards, axis=0)
    std = np.std(all_rewards, axis=0)
    x = np.arange(len(mean))
    plt.plot(x, mean, label=label, color=color)
    plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.3)

if __name__ == '__main__':
    env_name = 'Acrobot-v1'

    dqn_train_mean_rewards,dqn_test_mean_rewards = run_multiple_seeds(env_name, episodes=NUM_RUNS, steps=NUM_STEPS, ddqn_type='mean')
    dqn_train_max_rewards,dqn_test_max_rewards = run_multiple_seeds(env_name, episodes=NUM_RUNS, steps=NUM_STEPS, ddqn_type='max')

    plt.figure(figsize=(10, 6))
    plot_mean_std(dqn_train_mean_rewards, label='Dueling DQN (mean)', color='green')
    plot_mean_std(dqn_train_max_rewards, label='Dueling DQN (max)', color='blue')

    plt.title('Training Data Comparison of Mean vs Max Dueling DQN on Acrobot-v1 (Avg over 5 seeds)')
    plt.xlabel('Episode')
    plt.ylabel('Episodic Return (Smoothed)')
    plt.legend()
    plt.grid()
    plt.savefig('./plots/dueling_dqn_train_mean_vs_max_acrobot.png')
    plt.show()


    plt.figure(figsize=(10, 6))
    plot_mean_std(dqn_test_mean_rewards, label='Dueling DQN (mean)', color='green')
    plot_mean_std(dqn_test_max_rewards, label='Dueling DQN (max)', color='blue')

    plt.title('Testing Data Comparison of Mean vs Max Dueling DQN on Acrobot-v1 (Avg over 5 seeds)')
    plt.xlabel('Episode')
    plt.ylabel('Episodic Return (Smoothed)')
    plt.legend()
    plt.grid()
    plt.savefig('./plots/dueling_dqn_test_mean_vs_max_acrobot.png')
    plt.show()

