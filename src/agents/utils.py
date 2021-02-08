import random
import copy
import numpy as np
import torch

def centralize(states, actions):
    states = states.flatten(start_dim=1)
    actions = actions.flatten(start_dim=1)
    full_states = torch.cat([states, actions], dim=1)
    return full_states

def agent_batch_dim_swap(states, actions, rewards, next_states, dones):
    states = states.permute(1,0,-1)
    actions = actions.permute(1,0,-1)
    rewards = rewards.permute(1,0)
    next_states = next_states.permute(1,0,-1)
    dones = dones.permute(1,0) 
    return states, actions, rewards, next_states, dones

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        # dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        # dx = self.theta * (self.mu - x) + self.sigma * np.random.uniform(low=-1, high=1, size=self.mu.shape)
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(*self.mu.shape)
        self.state = x + dx
        return self.state

class GaussianNoise:
    def __init__(self, dim: int, mu=0, sigma=0.2):
        self.dim = dim
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        return np.random.normal(self.mu, self.sigma, self.dim)