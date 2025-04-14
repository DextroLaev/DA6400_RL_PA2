import tensorflow as tf
import numpy as np

from config import *


def PolicyModel(input_shape=(4,), output_shape=2):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(32, activation='selu')(inputs)
    x = tf.keras.layers.Dense(32, activation='selu')(x)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def BaselineModel(input_shape=(4,)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(32, activation='selu')(inputs)
    x = tf.keras.layers.Dense(32, activation='selu')(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


class TrajectoryBuffer:
    def __init__(self, num_envs=NUM_ENVS):
        self.num_envs = num_envs
        self.buffer = [[] for _ in range(num_envs)]

    def add(self, state, action, reward):
        if self.num_envs == 1:
            self.buffer[0].append((state, action, reward))
            return self.buffer
        for i in range(self.num_envs):
            self.buffer[i].append((state[i], action[i], reward[i]))
        return self.buffer

    def get_buffer(self, env_id):
        trajectory = self.buffer[env_id]
        states, actions, rewards = zip(*trajectory)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        return states, actions, rewards

    def clear(self, env_id):
        self.buffer[env_id] = []


class Agent:
    def __init__(self, input_shape=(4,), output_shape=2,type=None, num_envs=NUM_ENVS, learning_rate=ALPHA, gamma=GAMMA):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_envs = num_envs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.type = type

        if type == 'baseline':
            self.policy_model = PolicyModel(input_shape)
            self.baseline_model = BaselineModel(input_shape)
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate)
            self.baseline_optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate)
            self.policy_model.summary()
            self.baseline_model.summary()
        else:
            self.policy_model = PolicyModel(input_shape, output_shape)
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate)
            self.policy_model.summary()
        self.trajectory_buffer = TrajectoryBuffer(num_envs)

    @tf.function
    def _act(self, state):
        if state.ndim == 1:
            state = tf.expand_dims(state, axis=0)
        probs = self.policy_model(state)
        action = tf.random.categorical(tf.math.log(probs), num_samples=1)
        return tf.squeeze(action, axis=-1)

    def act(self, state, test=False):
        if self.num_envs == 1 or test==True:
            return self._act(state).numpy()[0]
        action = self._act(state)
        return action.numpy()

    @tf.function
    def learn(self, states, actions, returns):
        states = tf.cast(states, dtype=tf.float32)
        actions = tf.cast(actions, dtype=tf.int32)
        returns = tf.cast(returns, dtype=tf.float32)

        eps = 1e-8  # for log stability

        if self.type is None:
            with tf.GradientTape() as tape:
                probs = self.policy_model(states)
                action_one_hot = tf.one_hot(actions, self.output_shape)
                selected_probs = tf.reduce_sum(probs * action_one_hot, axis=1)
                log_probs = tf.math.log(selected_probs + eps)
                loss = -tf.reduce_mean(log_probs * returns)
            grads = tape.gradient(loss, self.policy_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.policy_model.trainable_variables))
            return loss
        else:
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                probs = self.policy_model(states)
                action_one_hot = tf.one_hot(actions, self.output_shape)
                selected_probs = tf.reduce_sum(probs * action_one_hot, axis=1)
                log_probs = tf.math.log(selected_probs + eps)

                baseline = tf.squeeze(self.baseline_model(states), axis=-1)
                advantage = returns - baseline

                actor_loss = -tf.reduce_mean(log_probs * advantage)
                critic_loss = tf.reduce_mean(tf.square(advantage))
                total_loss = actor_loss + critic_loss

            actor_grads = actor_tape.gradient(actor_loss, self.policy_model.trainable_variables)
            critic_grads = critic_tape.gradient(critic_loss, self.baseline_model.trainable_variables)

            self.optimizer.apply_gradients(zip(actor_grads, self.policy_model.trainable_variables))
            self.baseline_optimizer.apply_gradients(zip(critic_grads, self.baseline_model.trainable_variables))
            return total_loss

                

    def _compute_returns(self, rewards):
        returns = np.zeros_like(rewards)
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R
            returns[t] = R
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        return returns
