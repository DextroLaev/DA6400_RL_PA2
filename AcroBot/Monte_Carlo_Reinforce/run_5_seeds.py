# run_plot.py
import numpy as np
import matplotlib.pyplot as plt
from train import train_reinforce_mc
from config import *
import torch

def moving_average(rewards, window=100):
    return np.convolve(rewards, np.ones(window)/window, mode='valid')

def run_multiple_seeds(env_name, episodes, include_baseline=False, num_runs=5):
    all_runs = []
    for seed in range(num_runs):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f'Run {seed+1} Baseline={include_baseline}')
        rewards = train_reinforce_mc(env_name,seed, episodes, include_baseline=include_baseline,batch_size=BATCH_SIZE)
        all_runs.append(moving_average(rewards, window=20))
    return np.array(all_runs)


def plot_mean_std(all_rewards, label, color):
    mean = np.mean(all_rewards, axis=0)
    std = np.std(all_rewards, axis=0)
    x = np.arange(len(mean))
    plt.plot(x, mean, label=label, color=color)
    plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.3)

if __name__ == '__main__':
    env_name = 'Acrobot-v1'

    reinforce_baseline_rewards = run_multiple_seeds(env_name, episodes=NUM_RUNS, include_baseline=True, num_runs=5)
    reinforce_mc_rewards = run_multiple_seeds(env_name, episodes=NUM_RUNS, include_baseline=False, num_runs=5)

    plt.figure(figsize=(10, 6))
    plot_mean_std(reinforce_mc_rewards, label='Reinforce MC', color='green')
    plot_mean_std(reinforce_baseline_rewards, label='Reinforce MC with Baseline', color='blue')

    plt.xlabel('Episode')
    plt.ylabel('Episodic Return')
    plt.title('REINFORCE vs REINFORCE + Baseline on Acrobot-v1 (5 seeds)')
    plt.legend()
    plt.grid()
    plt.savefig('./plots/Reinforce_comparison_Monte_Acrobot.png')
    plt.show()