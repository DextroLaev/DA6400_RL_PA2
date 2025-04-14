import tensorflow as tf
import numpy as np

from config import *


def PolicyModel(input_shape=(6,), output_shape=3):
    initializer = tf.keras.initializers.LecunNormal()
    bias_initializer = tf.keras.initializers.Zeros()

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(
        32, activation='selu', kernel_initializer=initializer, bias_initializer=bias_initializer)(inputs)
    x = tf.keras.layers.Dense(
        32, activation='selu', kernel_initializer=initializer, bias_initializer=bias_initializer)(x)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax',
                                    kernel_initializer=initializer, bias_initializer=bias_initializer)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def BaselineModel(input_shape=(6,)):
    initializer = tf.keras.initializers.LecunNormal()
    bias_initializer = tf.keras.initializers.Zeros()

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(
        32, activation='selu', kernel_initializer=initializer, bias_initializer=bias_initializer)(inputs)
    x = tf.keras.layers.Dense(
        32, activation='selu', kernel_initializer=initializer, bias_initializer=bias_initializer)(x)
    outputs = tf.keras.layers.Dense(
        1, kernel_initializer=initializer, bias_initializer=bias_initializer)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


class TrajectoryBuffer:
    def __init__(self, num_envs=NUM_ENVS):
        self.num_envs = num_envs
        self.buffer = [[] for _ in range(num_envs)]

    def add(self, state, action, reward, next_state, done):
        if self.num_envs == 1:
            self.buffer[0].append((state, action, reward, next_state, done))
            return self.buffer
        for i in range(self.num_envs):
            self.buffer[i].append(
                (state[i], action[i], reward[i], next_state[i], done[i]))
        return self.buffer

    def get_buffer(self, env_id):
        trajectory = self.buffer[env_id]
        states, actions, rewards, next_states, dones = zip(*trajectory)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        return states, actions, rewards, next_states, dones

    def clear(self, env_id):
        self.buffer[env_id] = []


class Agent:
    def __init__(self, input_shape=(6,), output_shape=3, type=None, num_envs=NUM_ENVS, learning_rate=ALPHA, gamma=GAMMA):
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
            self.policy_model.summary()
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate)
        self.trajectory_buffer = TrajectoryBuffer(num_envs)

    @tf.function
    def _act(self, state):
        if state.ndim == 1:
            state = tf.expand_dims(state, axis=0)
        probs = self.policy_model(state)
        return probs

    def act(self, state, test=False):
        probs = self._act(state).numpy()  # Convert to NumPy
        if self.num_envs == 1 or test:
            action = np.random.choice(probs.shape[-1], p=probs[0])
            return action
        else:
            actions = [np.random.choice(p.shape[-1], p=p) for p in probs]
            return np.array(actions)

    @tf.function
    def learn_without_baseline(self, states, actions, returns):
        states = tf.cast(states, dtype=tf.float32)
        actions = tf.cast(actions, dtype=tf.int32)
        returns = tf.cast(returns, dtype=tf.float32)
        eps = 1e-8  # for log stability
        entropy_coeff = 0.01  # exploration encouragement

        with tf.GradientTape() as tape:
            probs = self.policy_model(states)
            action_one_hot = tf.one_hot(actions, self.output_shape)
            selected_probs = tf.reduce_sum(probs * action_one_hot, axis=1)
            log_probs = tf.math.log(selected_probs + eps)

            entropy = -tf.reduce_sum(probs * tf.math.log(probs + eps), axis=1)
            entropy_bonus = tf.reduce_mean(entropy)

            policy_loss = -tf.reduce_mean(log_probs * returns)
            total_loss = policy_loss - entropy_coeff * entropy_bonus

        grads = tape.gradient(
            total_loss, self.policy_model.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        self.optimizer.apply_gradients(
            zip(grads, self.policy_model.trainable_variables))
        return total_loss

    @tf.function
    def learn_with_baseline(self, states, actions, rewards, returns, next_states, dones):
        states = tf.cast(states, dtype=tf.float32)
        actions = tf.cast(actions, dtype=tf.int32)
        rewards = tf.cast(rewards, dtype=tf.float32)
        returns = tf.cast(returns, dtype=tf.float32)
        next_states = tf.cast(next_states, dtype=tf.float32)
        dones = tf.cast(dones, dtype=tf.float32)

        eps = 1e-8
        entropy_coeff = 0.01

        with tf.GradientTape() as policy_tape, tf.GradientTape() as baseline_tape:
            probs = self.policy_model(states)
            action_one_hot = tf.one_hot(actions, self.output_shape)
            selected_probs = tf.reduce_sum(probs * action_one_hot, axis=1)
            log_probs = tf.math.log(selected_probs + eps)

            entropy = -tf.reduce_sum(probs * tf.math.log(probs + eps), axis=1)
            entropy_bonus = tf.reduce_mean(entropy)

            baseline = tf.squeeze(self.baseline_model(states), axis=-1)
            advantage = returns - baseline

            policy_loss = -tf.reduce_mean(log_probs * advantage)
            total_policy_loss = policy_loss - entropy_coeff * entropy_bonus

            target_baseline = rewards + self.gamma * \
                (1.0 - dones) * tf.squeeze(self.baseline_model(next_states), axis=-1)
            target_baseline = tf.stop_gradient(target_baseline)
            baseline_loss = tf.reduce_mean(
                tf.square(target_baseline - baseline))

        policy_grads = policy_tape.gradient(
            total_policy_loss, self.policy_model.trainable_variables)
        policy_grads = [tf.clip_by_norm(g, 1.0) for g in policy_grads]
        self.optimizer.apply_gradients(
            zip(policy_grads, self.policy_model.trainable_variables))

        baseline_grads = baseline_tape.gradient(
            baseline_loss, self.baseline_model.trainable_variables)
        baseline_grads = [tf.clip_by_norm(g, 1.0) for g in baseline_grads]
        self.baseline_optimizer.apply_gradients(
            zip(baseline_grads, self.baseline_model.trainable_variables))

        return total_policy_loss + baseline_loss

    def _compute_returns(self, rewards):
        returns = np.zeros_like(rewards)
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R
            returns[t] = R
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        return returns
