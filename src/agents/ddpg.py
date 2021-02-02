import numpy as np
import random

import torch
import torch.optim as optim

from .model import Actor, Critic
from .utils import OUNoise
from ..utils.utils import DEVICE


class DDPG():
    def __init__(self, state_size, action_size, lr_actor, lr_critic, weight_decay_critic, random_seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor w/ target
        self.actor_local = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, seed=random_seed).to(DEVICE)
        self.actor_opt = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
    
        # Critic w/ target 
        self.critic_local = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
        self.critic_target = Critic(state_size, action_size, seed=random_seed).to(DEVICE)
        self.critic_opt = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay_critic)

        # ensure local and target networks start with same weights
        self._soft_update(self.actor_local, self.actor_target, 1)
        self._soft_update(self.critic_local, self.critic_target, 1)

        self.noise = OUNoise(action_size, random_seed)

    
    def act(self, state, use_target=False, noise_scale=0.0):
        state = torch.from_numpy(state).float().to(DEVICE)

        actor = self.actor_local if not use_target else self.actor_target
        actor.eval()
        with torch.no_grad():
            action = actor(state).cpu().data.numpy()
        actor.train()
        
        action += noise_scale * self.noise.sample()
        return np.clip(action, -1, +1)

    @staticmethod
    def _soft_update(local_model, target_model, tau):
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            mixed_param = tau * local_param.data + (1-tau)*target_param.data
            target_param.data.copy_(mixed_param)