import torch
import torch.nn.functional as F

from ..utils.utils import DEVICE
from .ddpg import DDPG
from .replay import ReplayBuffer
from .utils import agent_batch_dim_swap, centralize 


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 32        # minibatch size
NUM_BATCH = 1
GAMMA = 0.95            # discount factor
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
TAU = 1e-3              # for soft update of target parameters
WEIGHT_DECAY = 0        # L2 weight decay
TRAIN_FREQ = 20         # update net work every this many time steps

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

    def act(self, states, use_target=False, use_noise=True):
        actions = [agent.act(state, use_target, use_noise) for agent, state in zip(self.agents, states)]
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)
        if (self.t % TRAIN_FREQ == 0) & (len(self.memory) >= BATCH_SIZE * NUM_BATCH):
            experiences = self.memory.sample(NUM_BATCH)
            self._learn(experiences, GAMMA)
        self.t += 1
    
    def _learn(self, experiences, gamma):        
        for states, actions, rewards, next_states, dones in zip(*[torch.split(tensor, BATCH_SIZE) for tensor in experiences]):
            states, actions, rewards, next_states, dones = agent_batch_dim_swap(states, actions, rewards, next_states, dones)
            for agent, state, action, reward, next_state, done in zip(self.agents, states, actions, rewards, next_states, dones):
                # update critic
                next_actions = torch.tensor(self.act(next_states, use_target=True, use_noise=False)).to(DEVICE)
                Q_target_nexts = agent.critic_target(next_states, next_actions)
                Q_target = reward + gamma * (1 - done) * Q_target_nexts

                Q_current = agent.critic_target(states, actions)

                critic_loss = F.mse(Q_current, Q_target)
                agent.critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(),1)
                agent.critic_opt.step()

                # update actor

    def reset(self):
        [agent.noise.reset() for agent in self.agents]

