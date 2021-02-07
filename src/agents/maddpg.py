import torch
import torch.nn.functional as F

from ..utils.utils import DEVICE
from .ddpg import DDPG
from .replay import ReplayBuffer
from .utils import agent_batch_dim_swap, centralize 


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
NUM_BATCH = 2
GAMMA = 0.99            # discount factor
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
TAU = 1e-2              # for soft update of target parameters
WEIGHT_DECAY = 0        # L2 weight decay
TRAIN_FREQ = 1         # update net work every this many time steps

NOISE_DECAY = 0.995

class MADDPG():
    def __init__(self, state_size, action_size, num_agents, seed=37):
        self.agents = [
            DDPG(
                state_size = state_size,
                action_size = action_size,
                num_agents = num_agents,
                lr_actor=LR_ACTOR,
                lr_critic=LR_CRITIC,
                weight_decay_critic=WEIGHT_DECAY,
                random_seed=seed
            )
            for _ in range(num_agents)
        ]
        self.memory = ReplayBuffer(
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            seed=seed
        )

        self.t = 0
        self.noise_scale = 1

    def decay_noise(self, noise_decay=NOISE_DECAY):
        self.noise_scale *= noise_decay

    def act(self, states, use_target=False, use_noise=True):
        actions = [agent.act(state, use_target, use_noise, self.noise_scale) for agent, state in zip(self.agents, states)]
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)
        if (self.t % TRAIN_FREQ == 0) & (len(self.memory) >= BATCH_SIZE * NUM_BATCH):
            experiences = self.memory.sample(NUM_BATCH)
            self._learn(experiences, GAMMA)
        
        if False not in dones:
            self.decay_noise()
        self.t += 1
    
    def _learn(self, experiences, gamma):        
        for states, actions, rewards, next_states, dones in zip(*[torch.split(tensor, BATCH_SIZE) for tensor in experiences]):
            states, actions, rewards, next_states, dones = agent_batch_dim_swap(states, actions, rewards, next_states, dones)
            for agent, state, action, reward, next_state, done in zip(self.agents, states, actions, rewards, next_states, dones):
                # update critic
                next_actions = torch.tensor(self.act(next_states, use_target=True, use_noise=False)).to(DEVICE)
                Q_target_nexts = agent.critic_target(next_states, next_actions).squeeze()
                Q_target = reward + gamma * (1 - done) * Q_target_nexts

                Q_current = agent.critic_local(states, actions).squeeze()

                # critic_loss = F.mse_loss(Q_current, Q_target.detach())
                critic_loss = F.smooth_l1_loss(Q_current, Q_target.detach())
                agent.critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
                agent.critic_opt.step()

                # update actor
                actions_pred = torch.tensor(self.act(states, use_noise=False)).to(DEVICE)
                actor_loss = -agent.critic_local(states, actions_pred).mean()
                agent.actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor_local.parameters(), 1)
                agent.actor_opt.step()
            self._network_update(TAU)

    def reset(self):
        [agent.reset() for agent in self.agents]
    
    def _network_update(self, tau):
        [agent._network_update(agent.critic_local, agent.critic_target, tau) for agent in self.agents]
        [agent._network_update(agent.actor_local, agent.actor_target, tau) for agent in self.agents]

    def save(self, fp):
        models = {}
        for i, agent in enumerate(self.agents):
            models[f'agent_{i}_actor'] = agent.actor_local.state_dict()
            models[f'agent_{i}_critic'] = agent.critic_local.state_dict()
        torch.save(models, fp)

