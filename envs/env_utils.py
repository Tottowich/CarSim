# This is a utils file ofr enviromental functions when using pytorch for reinforcement learning
# The enviroments are created using the gym library from OpenAI
#
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np

# This is a parent class for all the enviroments
class Env:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.state = self.env.reset()
        self.done = False
        self.reward = 0
        self.info = None
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        self.state = self.env.reset()
        self.done = False
        self.reward = 0
        self.info = None
        return self.state

    def step(self, action):
        self.state, self.reward, self.done, self.info = self.env.step(action)
        return self.state, self.reward, self.done, self.info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)
