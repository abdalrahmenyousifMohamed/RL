# Main code

from __future__ import print_function
import os
import torch
import warnings

warnings.simplefilter("ignore")

import torch.multiprocessing as mp
from envs import create_atari_env
from model import ActorCritic
from train import train
from test import test
import my_optim

# Gathering all the parameters (that we can modify to explore)
class Params():
    def __init__(self):
        self.lr = 0.0001
        self.gamma = 0.99
        self.tau = 1.
        self.seed = 1
        self.num_processes = 16
        self.num_steps = 20
        self.max_episode_length = 10000
        self.env_name = 'Breakout-v0'

# Main run
os.environ['OMP_NUM_THREADS'] = '1' # 1 thread per core
params = Params() # creating the params object from the Params class, that sets all the model parameters
torch.manual_seed(params.seed) # setting the seed (not essential)
env = create_atari_env(params.env_name) # we create an optimized environment thanks to universe
shared_model = ActorCritic(env.observation_space.shape[0], env.action_space) # shared_model is the model shared by the different agents (different threads in different cores)
shared_model.share_memory() # storing the model in the shared memory of the computer, which allows the threads to have access to this shared memory even if they are in different cores
optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=params.lr) # the optimizer is also shared because it acts on the shared model
optimizer.share_memory() # same, we store the optimizer in the shared memory so that all the agents can have access to this shared memory to optimize the model

if __name__ == '__main__':
    # create the processes
    mp.freeze_support() # call this before creating any processes
    processes = []
    p = mp.Process(target=test, args=(params.num_processes, params, shared_model))
    p.start()
    processes.append(p)
    for rank in range(0, params.num_processes):
        p = mp.Process(target=train, args=(rank, params, shared_model, optimizer))
        p.start()
        processes.append(p)

    # wait for the processes to finish
    for p in processes:
        p.join()
