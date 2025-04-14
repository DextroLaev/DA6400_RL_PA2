from agent import Agent
from config import *
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt

def moving_average(rewards, window=100):
    return np.convolve(rewards, np.ones(window)/window, mode='valid')

def train_reinforce_mc(env_name,seed, epsiodes, batch_size, save_model=False, include_baseline=False):
    np.random.seed(seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed)
    env = gym.make(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    n_actions = env.action_space.n
    input_dims = env.observation_space.shape[0]

    agent = Agent(gamma=GAMMA, input_dims=input_dims, lr_policy=ALPHA_POLICY,lr_value=ALPHA_VALUE,
                  output_dims=n_actions, include_baseline=include_baseline)

    all_rewards = []
    trajectory_batch = []

    for episode in range(1,epsiodes+1):
        state, _ = env.reset()

        log_probs = []
        rewards = []
        old_states,next_states = [],[]
        dones= []
        done = False
        score = 0

        for step in range(MAX_STEPS):
            action,log_prob = agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            old_states.append(state)
            next_states.append(next_state)
            dones.append(done)
            state = next_state
            score += reward
            if done or truncated:
                break

        trajectory_batch.append((log_probs, rewards,old_states,next_states,dones))
        all_rewards.append(score)
        avg_score = np.mean(all_rewards[-50:])
                
        
        if (episode) % batch_size == 0:
            agent.learn(trajectory_batch)
            trajectory_batch = []

        print(f'\rEpisode {episode}/{epsiodes} | '
                  f'Score: {score:.1f} | Avg Score (last 50): {avg_score:.2f}',end='',flush=True)

    print()
    if save_model:
        model_path = f'./models/reinforce_monte_Carlo_baseline_{include_baseline}_acrobot.pt'
        torch.save({
            'model_state_dict': agent.policy.state_dict(),
            'optimizer_state_dict': agent.policy.optimizer.state_dict()
        }, model_path)
        print(f"Model saved to {model_path}")
    
    return all_rewards

if __name__ == '__main__':
    scores_baseline = train_reinforce_mc('Acrobot-v1',seed=123,epsiodes=NUM_RUNS,save_model=True,include_baseline=True,batch_size=BATCH_SIZE)
    scores_nobaseline = train_reinforce_mc('Acrobot-v1',seed=2024,epsiodes=NUM_RUNS,save_model=True,include_baseline=False,batch_size=BATCH_SIZE)
    plt.plot(moving_average(scores_baseline), label='With Baseline')
    plt.plot(moving_average(scores_nobaseline), label='Without Baseline')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('REINFORCE with vs without Baseline')
    plt.legend()
    plt.grid()
    plt.show()
