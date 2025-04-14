import torch

NUM_RUNS = 3000
GAMMA = 0.99
ALPHA_POLICY = 1e-3
ALPHA_VALUE = 5e-6
MAX_STEPS = 1000
BATCH_SIZE=10
GRAD_MAX_NORM = 10

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")