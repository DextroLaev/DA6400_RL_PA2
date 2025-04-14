import tensorflow as tf
import numpy as np

from config import *

def QModelMean(input_shape=(4,), output_shape=2):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(32, activation='selu')(inputs)
    x = tf.keras.layers.Dense(32, activation='selu')(x)

    value = tf.keras.layers.Dense(32, activation='selu')(x)
    value = tf.keras.layers.Dense(1)(value)
    advantage = tf.keras.layers.Dense(32, activation='selu')(x)
    advantage = tf.keras.layers.Dense(output_shape)(advantage)
    def combine_q(inputs):
        value, advantage = inputs
        advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
        return value + (advantage - advantage_mean)
    q_values = tf.keras.layers.Lambda(combine_q)([value, advantage])
    model = tf.keras.Model(inputs=inputs, outputs=q_values)
    return model

def QModelMax(input_shape=(4,), output_shape=2):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(32, activation='selu')(inputs)
    x = tf.keras.layers.Dense(32, activation='selu')(x)

    value = tf.keras.layers.Dense(32, activation='selu')(x)
    value = tf.keras.layers.Dense(1)(value)
    advantage = tf.keras.layers.Dense(32, activation='selu')(x)
    advantage = tf.keras.layers.Dense(output_shape)(advantage)
    def combine_q(inputs):
        value, advantage = inputs
        advantage_mean = tf.reduce_max(advantage, axis=1, keepdims=True)
        return value + (advantage - advantage_mean)
    q_values = tf.keras.layers.Lambda(combine_q)([value, advantage])
    model = tf.keras.Model(inputs=inputs, outputs=q_values)
    return model

class ReplayBuffer:
    def __init__(self,buffer_size=BUFFER_SIZE,num_envs=NUM_ENVS):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.buffer = []
        self.index = 0
        
    def __len__(self):
        return len(self.buffer)
        
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.index] = (state, action, reward, next_state, done)
            self.index = (self.index + 1) % self.buffer_size
        
    def sample(self, batch_size=NUM_BATCH_SIZE):
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough samples in the buffer to sample from.")
        indices = np.random.choice(len(self.buffer), size=int(batch_size/self.num_envs), replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        if self.num_envs > 1:
            states = np.concatenate(states, axis=0)
            actions = np.concatenate(actions, axis=0)
            rewards = np.concatenate(rewards, axis=0)
            next_states = np.concatenate(next_states, axis=0)
            dones = np.concatenate(dones, axis=0)
        else:
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)
        return states, actions, rewards, next_states, dones


class Agent:
    def __init__(self, input_shape, output_shape,type='mean', num_envs=NUM_ENVS, buffer_size=BUFFER_SIZE, learning_rate=ALPHA,gamma=GAMMA, tau=TAU):
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        if type == 'mean':
            self.model = QModelMean(input_shape=self.input_shape, output_shape=self.output_shape)
            self.target_model = QModelMean(input_shape=self.input_shape, output_shape=self.output_shape)
        elif type == 'max':
            self.model = QModelMax(input_shape=self.input_shape, output_shape=self.output_shape)
            self.target_model = QModelMax(input_shape=self.input_shape, output_shape=self.output_shape)
        else:
            raise ValueError(f"Unknown model type: {type}")
        self.target_model.set_weights(self.model.get_weights())
        self.model.summary()
        self.replay_buffer = ReplayBuffer(buffer_size=self.buffer_size, num_envs=self.num_envs)
        
        self.epsilon_start = self.epsilon = 1.0
        self.epsilon_end = 0.05
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA)
    
    @tf.function
    def _act(self, state):
        if state.ndim == 1:
            state = tf.expand_dims(state, axis=0)
        values = self.model(state)
        action = tf.argmax(values, axis=1)
        return action
    
    def epsilon_value(self,step, num_steps):
        return max(self.epsilon_end, self.epsilon_start * (1 - (step / num_steps)))
        
    
    def act(self, state, epsilon=None,test=False):
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.rand() < epsilon:
            if self.num_envs == 1 or test == True:
                return np.random.randint(low=0, high=self.output_shape)
            return np.random.randint(low=0, high=self.output_shape, size=(self.num_envs,))
        else:
            if self.num_envs == 1 or test == True:
                return self._act(state).numpy()[0]
            return self._act(state).numpy()
    
    @tf.function
    def update(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        actions = tf.cast(actions, tf.int32)
        rewards = tf.cast(rewards, tf.float32)
        dones = tf.cast(dones, tf.float32)
        
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_values_selected = tf.reduce_sum(q_values * tf.one_hot(actions, depth=self.output_shape), axis=1)
            q_values_next = self.target_model(next_states)
            max_q_next = tf.reduce_max(q_values_next, axis=1)
            targets = rewards + (1.0 - dones) * self.gamma * max_q_next
            targets = tf.stop_gradient(targets)
            loss = tf.reduce_mean(tf.square(q_values_selected - targets))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

        
    def update_target(self):
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_weights[i] = (1 - self.tau) * target_weights[i] + self.tau * model_weights[i]
        self.target_model.set_weights(target_weights)
            
if __name__ == '__main__':
    state = np.random.randn(NUM_ENVS,4)
    agent = Agent(input_shape=(4,),output_shape=2, num_envs=NUM_ENVS)
    action = agent.act(state)