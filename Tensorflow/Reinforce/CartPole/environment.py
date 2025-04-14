import gymnasium as gym
import numpy as np


class Environment:
    def __init__(self, name='CartPole-v1', num_envs=10):
        self.name = name
        self.num_envs = num_envs
        if self.num_envs > 1:
            self.envs = gym.make_vec(
                name, num_envs=self.num_envs, vectorization_mode='async')
        else:
            self.envs = gym.make(name, render_mode='human')
            self.envs = gym.wrappers.Autoreset(self.envs)

    def reset(self):
        return self.envs.reset()[0]

    def step(self, actions):
        return self.envs.step(actions)


def simulate(steps=1000):
    env = Environment(num_envs=8)
    state = env.reset()
    for _ in range(steps):
        actions = np.random.randint(low=0, high=2, size=(env.num_envs,))
        next_state, rewards, dones, trauncated, info = env.step(actions)
        state = next_state
        if np.any(dones):
            print(dones,state.shape)


if __name__ == '__main__':
    simulate()