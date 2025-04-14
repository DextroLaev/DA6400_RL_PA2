import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import sys

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


def train(environment, agent, epsilons, step, num_steps=NUM_TRAIN_STEPS):
    state = environment.reset()
    for _ in (range(0,num_steps)):
        current_step = _+step-1
        epsilon = epsilons[current_step]
        actions = agent.act(state, epsilon=epsilon)
        next_state, rewards, dones, truncated, info = environment.step(actions)
        agent.replay_buffer.add(state, actions, rewards, next_state, dones)
        state = next_state
        if len(agent.replay_buffer) > NUM_BATCH_SIZE and _ % NUM_UPDATE_EVERY == 0:
            states, actions, rewards, next_states, dones = agent.replay_buffer.sample(batch_size=NUM_BATCH_SIZE)
            loss = agent.update((states, actions, rewards, next_states, dones))
            print(f"\rTraining step: [{current_step}/{NUM_TRAIN_STEPS}] \tLoss: {loss.numpy():.4f} \tEpsilon: {epsilon:.4f}",end='')
            sys.stdout.flush()
        if _ % NUM_UPDATE_TARGET_EVERY == 0:    
            agent.update_target()
    return agent

def test(environment, agent, num_steps=500):
    state = environment.reset()
    done = False
    rewards_collected = 0
    for _ in range(num_steps):
        actions = agent.act(state, epsilon=0.05,test=True)
        next_state, rewards, done, truncated, info = environment.step(actions)
        state = next_state
        rewards_collected += rewards
        if done:
            break
    return rewards_collected

def create_epsilons(NUM_STEPS, epsilon_start=1.0, epsilon_min=0.05, fraction=0.5):
    decay_steps = int(fraction * NUM_STEPS)
    linear_decay = np.linspace(epsilon_start, epsilon_min, decay_steps)
    constant_tail = np.full(NUM_STEPS - decay_steps, epsilon_min)
    epsilons = np.concatenate([linear_decay, constant_tail])
    return epsilons

def session(num_steps = NUM_TRAIN_STEPS,type='mean'):
    env = Environment(num_envs=NUM_ENVS)
    test_env = Environment(num_envs=1)
    agent = Agent(input_shape=(4,), output_shape=2, num_envs=NUM_ENVS, type=type)
    epsilons = create_epsilons(num_steps,fraction=0.5)
    num_steps_to_train_for = 1000
    
    avg_reward_collection = []
    for _ in tqdm.tqdm(range(1,num_steps+1, num_steps_to_train_for)):
        agent = train(environment=env,epsilons=epsilons,step=_, agent=agent, num_steps=num_steps_to_train_for)
        test_rewards = [test(environment=test_env, agent=agent, num_steps=500) for i in range(10)]
        avg_test_reward = np.mean(test_rewards)
        avg_reward_collection.append(avg_test_reward)
        print(f"\nAverage test reward: {avg_test_reward:.2f}")
    return avg_reward_collection
        
if __name__ == '__main__':
    history_mean = [session(num_steps=NUM_TRAIN_STEPS, type='mean') for i in range(5)]
    history_mean = np.array(history_mean)
    avg_history_mean = np.mean(history_mean, axis=0)
    std_history_mean = np.std(history_mean, axis=0)
    
    history_max = [session(num_steps=NUM_TRAIN_STEPS, type='max') for i in range(5)]
    history_max = np.array(history_max)
    avg_history_max = np.mean(history_max, axis=0)
    std_history_max = np.std(history_max, axis=0)
    
    plt.figure(figsize=(10,5))
    
    plt.plot(avg_history_mean, label='Dueling Mean')
    plt.fill_between(np.arange(len(avg_history_mean)), avg_history_mean-std_history_mean, avg_history_mean+std_history_mean, alpha=0.2)
    
    plt.plot(avg_history_max, label='Dueling Max')
    plt.fill_between(np.arange(len(avg_history_max)), avg_history_max-std_history_max, avg_history_max+std_history_max, alpha=0.2)
    
    plt.title('Average Test Reward over Training Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Average Test Reward')
    plt.legend()
    plt.grid()
    plt.savefig('training_results.png')
    # plt.show()
        
    