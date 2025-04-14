import torch
import gymnasium as gym
from agent import Agent
from config import *

def evaluate_dueling_dqn(env_name,runs, model_path, ddqn_type='mean', render_mode='human',include_baseline=True):
    env = gym.make(env_name, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = Agent(gamma=GAMMA, input_dims=obs_dim, lr_policy=ALPHA_POLICY,lr_value=ALPHA_VALUE,
                  output_dims=n_actions, include_baseline=include_baseline)

    checkpoint = torch.load(model_path, map_location=DEVICE,weights_only=True)
    agent.policy.load_state_dict(checkpoint['model_state_dict'])
    agent.policy.eval()

    for i in range(runs):
        observation, _ = env.reset()
        done = False
        score = 0
        steps = 0
        while not done:
            action,log = agent.choose_action(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            steps += 1
        print("Total reward: {} Total steps: {}".format(score,steps))
    env.close()

if __name__ == '__main__':
    include_baseline = False
    model_path = './models/reinforce_monte_Carlo_baseline_{}_acrobot.pt'.format(include_baseline)
    evaluate_dueling_dqn('Acrobot-v1',5,model_path,ddqn_type='mean',render_mode='human',include_baseline=include_baseline)
