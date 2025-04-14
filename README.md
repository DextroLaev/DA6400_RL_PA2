# Reinforcement Learning PA2: Dueling DQN and Policy Gradient (and Reinforce) for CartPole and Acrobot

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.4.1-orange)](https://github.com/google/jax)
[![NumPy](https://img.shields.io/badge/NumPy-1.22.0-blue)](https://numpy.org/)

This repository contains the implementation and analysis of Q-Learning and SARSA algorithms applied to two classic reinforcement learning environments: CartPole-v1 and Acrobot-v1. The project was completed as part of the DA6400 Reinforcement Learning course.

## Project Overview
This project empirically evaluates two fundamental Reinforcement Learning (RL) algorithms—SARSA and Q-Learning—across three Gymnasium environments with varying complexity:
- **CartPole-v1**: Balance a pole on a moving cart
- **Acrobot-v1**: Drive a car up a steep hill using momentum

Key contributions include:
- Implementation of both Dueling DQN and Reinforce approaches
- Comparative analysis of exploration strategies: ε-greedy and Monte Carlo
- Hyperparameter tuning for optimal performance
- Robust evaluation across 5 random seeds

## Algorithms Implemented

### REINFORCE (Monte Carlo Policy Gradient)
```python
θ ← θ + α ∇θ log π_θ(a | s) · G_t
```
- Uses stochastic policy (Softmax or Gaussian)  
- Updates parameters using returns from full episodes  
- Can include baseline to reduce variance  

### DQN (Deep Q-Network)
```python
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s', a') - Q(s,a)]
```
- Uses ε-greedy exploration policy  
- Experience replay and target network for stability  
- Learns value function with neural network approximation  


### Prerequisites
- Python 3.8+
- Tensorflow
- Torch
- Gymnasium
- NumPy

### How to run:
```bash
cd Tensorflow
cd Reinforce
cd Cartpole
python3 train_and_test.py # or
bash run.sh
```

## Team Members
- Shuvrajeet Das [DA24D402] (IIT Madras, DSAI Dept)  
  [da24d402@smail.iitm.ac.in] | [shuvrajeet17@gmail.com]

- Rajshekhar Rakshit [CS24S031]  (IIT Madras, CSE Dept)  
  [cs24s031@smail.iitm.ac.in] | [rajshekharrakshit123@gmail.com]

## References
1. Barto, Sutton, Anderson (1983) - Neuronlike adaptive elements
2. Moore (1990) - Efficient memory-based learning for robot control
