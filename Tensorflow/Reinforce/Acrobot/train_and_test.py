import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import sys
import gc

from config import *
from agent import *
from environment import *

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def train(environment, agent, step, num_steps=NUM_TRAIN_STEPS):
    state = environment.reset()
    for _ in (range(num_steps)):
        current_step = _+step-1
        action = agent.act(state)
        next_state, reward, done, truncated, info = environment.step(action)
        done = np.logical_or(done, truncated)
        # print(done,action,truncated)
        # if np.all(done):
        #     exit()
        agent.trajectory_buffer.add(state, action, reward, next_state, done)
        if np.any(done):
            if agent.trajectory_buffer.num_envs == 1:
                states, actions, rewards, next_states, dones = agent.trajectory_buffer.get_buffer(
                    0)
                returns = agent._compute_returns(rewards)
                if agent.type == None:
                    loss = agent.learn_without_baseline(states, actions, returns)
                else:
                    loss = agent.learn_with_baseline(states, actions, rewards, returns, next_states, dones)
                print(
                    f"\rTraining step: [{current_step}/{num_steps}] \tLoss: {loss.numpy():.4f}", end='')
                sys.stdout.flush()
                agent.trajectory_buffer.clear(0)
            else:
                for i in range(environment.num_envs):
                    if done[i]:
                        states, actions, rewards, next_states, dones = agent.trajectory_buffer.get_buffer(i)
                        returns = agent._compute_returns(rewards)
                        if agent.type == None:
                            loss = agent.learn_without_baseline(states, actions, returns)
                        else:
                            loss = agent.learn_with_baseline(states, actions, rewards, returns, next_states, dones)
                        print(
                            f"\rTraining step: [{current_step}/{NUM_TRAIN_STEPS}] \tLoss: {loss.numpy():.4f}", end='')
                        sys.stdout.flush()
                        agent.trajectory_buffer.clear(i)
        state = next_state
    return agent


def test(environment, agent, num_steps=500):
    state = environment.reset()
    done = False
    rewards_collected = 0
    for _ in range(num_steps):
        actions = agent.act(state, test=True)
        next_state, rewards, done, truncated, info = environment.step(actions)
        state = next_state
        rewards_collected += rewards
        if done:
            break
    return rewards_collected


def simulation(num_steps=NUM_TRAIN_STEPS, type=None):
    env = Environment(num_envs=NUM_ENVS)
    test_env = Environment(num_envs=1)
    agent = Agent(input_shape=(6,), output_shape=3,
                  num_envs=NUM_ENVS, type=type)
    num_steps_to_train_for = 1000

    avg_reward_collection = []
    for _ in tqdm.tqdm(range(1, num_steps+1, num_steps_to_train_for)):
        agent = train(
            env, agent, _, num_steps=num_steps_to_train_for)
        test_rewards = [test(test_env, agent, num_steps=500)
                        for i in range(10)]
        avg_reward = np.mean(test_rewards)
        avg_reward_collection.append(avg_reward)
        print(f"\nAverage test reward: {avg_reward:.2f}")
        gc.collect()
    return avg_reward_collection


if __name__ == '__main__':
    
    history_baseline = [simulation(
        num_steps=NUM_TRAIN_STEPS, type='baseline') for i in range(5)]
    history_baseline = np.array(history_baseline)
    history_baseline_mean = np.mean(history_baseline, axis=0)
    history_baseline_std = np.std(history_baseline, axis=0)
    
    history = [simulation(num_steps=NUM_TRAIN_STEPS,
                          type=None) for i in range(5)]
    history = np.array(history)
    history_mean = np.mean(history, axis=0)
    history_std = np.std(history, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(history_mean, label='Without Baseline')
    plt.fill_between(np.arange(len(history_mean)), history_mean-history_std,
                     history_mean+history_std, alpha=0.2)
    plt.plot(history_baseline_mean, label='Baseline')
    plt.fill_between(np.arange(len(history_baseline_mean)), history_baseline_mean-history_baseline_std,
                     history_baseline_mean+history_baseline_std, alpha=0.2)
    plt.title('Average Test Reward')
    plt.xlabel('Training Steps')
    plt.ylabel('Average Test Reward')
    plt.legend()
    plt.grid()
    plt.savefig('training_results.png')
    plt.show()
