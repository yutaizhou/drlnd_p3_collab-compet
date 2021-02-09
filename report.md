## Learning Algorithm

### Agent Architecture
The **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** algorithm as described in the [2017 paper](maddpg_paper) is used to solve the task. It is an actor-critic algoritm applied in the multi-agent setting, and it has two types of deep neural networks (DNN) for each involved agent, one to represent the actor, and one for the critic. 
In this case, the critic variant is denoted as a centralized critic, as it takes in information from all agents in the environment during training, which is impossible to do during evaluation. The actor, on the other hand, only takes in information from the agent it belongs to. Hence, the MADDPG algorithm adopts a centralized training, decentralized execution (CTDE) strategy.

The DNN used to approximation the Q function (centralized critic) is a simple 3 layer fully connected feedforward neural network. Since the state space, |S| = 24, the action space, |A| = 2, and there are 2 agents, the input size of the first layer is (24 + 2) * 2 = 52, The output size of the network is 1, since the centralized critic simply predicts the scalar Q-value of the given state and joint action from all agents. 

The DNN used to approximation the policy (critic) is a simple 3 layer fully connected feedforward neural network. It has an input size of |S| = 24, and an output size of |A| = 2. tanh function is used at the output layer to squash the logits into the range [-1, +1] since the action space is continuous in the range [-1, +1]

Instead of the Ornstein–Uhlenbeck Process proposed in the original paper, it was empirically found that simple Gaussian noise was enough to perturb the action space to encourage exploration. A noise scaling constant was introduced to encourage exploration in the earlier episodes, and to encourage exploitation in the later episodes.

### MADDPG: Extending DDPG to Multi-Agent Settings
Although Deep Deterministic Policy Gradient (DDPG) is introduced in the [2015 paper](ddpg_paper) as an actor-critic algorithm, it is widely thought of as a continuous action space extension of the [DQN](dqn_paper) algorithm. For more information on DDPG theory, please refer to the [report](p2) of the second project. 

Many options are possible for extending DDPG into the multi-agent setting. The following are some of the most prominent ones. Note that the extensions apply to any single-agent RL algorithms, not just DDPG.

1. __Independent learning__: Have every agent use the original DDPG algorithm, where each agent naively treats all other agents as part of the environment.
2. __Centralized Training Centralized Execution__: Learn a joint policy that simultaneously outputs the actions for all agents. 
3. __Centralized Training Decentraliezd Execution__: Each agent learns and uses its own policy which only requires that agent's information, but during training, extra information not available during evaluation may be used to train the policy and critic. 

__Approach 1__ is the most naive and simplest case, but it also suffers from the nonstationarity problem: since environment distribution includes other agents, which are entities that learn and change their behavior overtime, any agent must continuously adjust its policy to handle an ever-changing environment distribution. 

__Approach 2__ addressed the nonstationarity problem since the factors that would otherwise cause nonstationarity are taken account for, but the joint policy responsible for centralized execution has an action space that exponentially scales with the number of agents. E.g. in the tennis environment with |A|=2, joint policy with one agent would have |A|=2, two agents |A|=4, three agents |A|=8, etc. 

__Approach 3__ is an intermediary between 1 and 2. Those extra information typically comes in the form of observations and actions taken by other agents, which, in lieu of perfect communication or full observability, would not be avaiable to an agent during evaluation. To utilize those information, a centralized critic is typically used. Whereas a critic in a single agent algorithm must output the expected return for a given state or state-action pair from one agent, a centralized critic outputs the expected return given observations and actions from all agents. 

__Parameter Sharing__ is yet another technique that can be used in all of the above approaches. It is where the parameters of the actor and/or critic between agents is shared/copied. In MADDPG, each agent learns its own centralized critic and actor, so no parameter sharing is done. In practice, parameter sharing combined with independent learning, i.e., all agents use and update the same networks for their actor and/or critic, works embarassing well. 


_Hyperparameters_
- BUFFER_SIZE = int(1e6)  # replay buffer size
- BATCH_SIZE = 256        # minibatch size
- NUM_BATCH = 1           # number of batches to sample/train on during each update iteration
- GAMMA = 0.99            # discount factor
- LR_ACTOR = 1e-4         # learning rate of the actor 
- LR_CRITIC = 1e-3        # learning rate of the critic
- TAU = 1e-3              # for soft update of target parameters
- WEIGHT_DECAY = 0        # L2 weight decay
- TRAIN_FREQ = 1          # update net work after this many time steps
- NOSIE_DECAY = 0.995     # exponential decay on noise scaling 

## Reward Plot

![Reward Plot][reward_plot]

The task was solved in 5032 episodes.


## Future Work

MADDPG could be combined with parameter sharing, more specifically, the centralized critic could be shared between all agents. 

MADDPG is an actor-critic off-policy algorithm. [SAC](sac_paper) is yet another algorithm of this type that can be easily extended to the multiagent setting using centralized critic and parameter sharing. [PPO](ppo_paper) is a policy-based on-policy algorithm that can also be easily extended using similar means. 

Attention combined with multi-agent RL is an interesting approach, as it uses the attention mechanism to reason about which agent to pay more attention to. E.g. Is agent 3 visiting a rare state? Use attention to give its value contribution higher weighting! [MAAC]([attn_paper]) explores this concept. 

<!-- Links -->
[reward_plot]: https://github.com/yutaizhou/drlnd_p3_collab-compet/blob/main/results/MADDPG/result.png
[maddpg_paper]: https://arxiv.org/abs/1706.02275
[ddpg_paper]: https://arxiv.org/abs/1509.02971
[dqn_paper]: https://www.nature.com/articles/nature14236
[sac_paper]: https://arxiv.org/abs/1801.01290
[ppo_paper]: https://arxiv.org/abs/1707.06347
[attn_paper]: https://arxiv.org/abs/1810.02912
[p2]: https://github.com/yutaizhou/drlnd_p2_continuous_control/blob/main/report.md
