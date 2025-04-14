import torch
import gymnasium as gym
from agent import Agent
from config import *

def evaluate_dueling_dqn(env_name,runs, model_path, ddqn_type='mean', render_mode='human'):
    env = gym.make(env_name, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = Agent(gamma=GAMMA, epsilon=0.0, input_dims=obs_dim, batch_size=BATCH_SIZE, lr=ALPHA,
                  n_actions=n_actions, eps_end=EPSILON_END, max_mem_size=MEM_SIZE,
                  target_update_freq=TARGET_NET_FREQ, ddqn_type=ddqn_type)

    checkpoint = torch.load(model_path, map_location=agent.Q_network.device,weights_only=True)
    agent.Q_network.load_state_dict(checkpoint['model_state_dict'])
    agent.Q_network.eval()

    for i in range(runs):
        observation, _ = env.reset()
        done = False
        score = 0
        steps = 0
        while not done:
            action = agent.choose_action(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            steps += 1
        print("Total reward: {} Total steps: {}".format(score,steps))
    env.close()

if __name__ == '__main__':
    model_path = './models/dueling_dqn_mean_acrobot.pt'
    evaluate_dueling_dqn('Acrobot-v1',5,model_path,ddqn_type='max',render_mode='human')
