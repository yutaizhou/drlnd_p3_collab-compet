import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from unityagents import UnityEnvironment

from src.agents.maddpg import MADDPG
from src.utils.utils import Logger
from src.utils.typing import List


def rollout(agent, env: UnityEnvironment, is_training: bool = True):
    # completes one episode of rollout, is_training functionality not really fully fledged out
    env_info = env.reset(train_mode=True)[brain_name] 
    
    states = env_info.vector_observations
    total_reward = 0
    ended = False
    while not ended:
        actions = agent.act(states)

        env_info = env.step(actions)[brain_name]
        actions, next_states, rewards, dones = np.array(actions), env_info.vector_observations, np.array(env_info.rewards), np.array(env_info.local_done)
        ended = False not in dones
        agent.step(states, actions, rewards, next_states, dones)
        states = next_states
        total_reward += rewards

    agent.decay_noise()
    return total_reward.max()


def run(agent, agent_name, env: UnityEnvironment, num_episodes=10000, is_training=True) -> List[float]:
    scores = []
    max_avg_score = -np.inf
    solved = False

    logger = Logger(f'results/{agent_name}/progress.txt')
    logger.write(f'Progress for {agent_name} agent\n')
    for i_episode in trange(1, num_episodes+1):
        total_reward = rollout(agent, env, is_training)
        scores.append(total_reward)

        if is_training:
            if len(scores) >= 100:
                avg_score = np.mean(scores[-100:])
                max_avg_score = max(max_avg_score, avg_score)
            
            if i_episode % 100 == 0:
                logger.write(f'Episode {i_episode}/{num_episodes} | Max Average Score: {max_avg_score}\n')
            if max_avg_score >= 0.5 and not solved:
                logger.write(f'Task solved in {i_episode} episodes, with average score over the last 100 episode: {max_avg_score}\n')
                solved = True
                agent.save(f'results/{agent_name}/checkpoint.pth')

    logger.close()
    with open(f'results/{agent_name}/scores.npy', 'wb') as f:
        np.save(f, scores)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a runner for DRLND Project 3: Collaboration and Competition')
    parser.add_argument('algorithm', type=str)
    args = parser.parse_args()

    # set up env
    env = UnityEnvironment(file_name="src/envs/Tennis_Linux_NoVis/Tennis.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]


    # set up agent
    num_agents, state_size = env_info.vector_observations.shape
    action_size = brain.vector_action_space_size

    algorithms = {'MADDPG': MADDPG}
    algorithm = algorithms[args.algorithm]
    agent = algorithm(state_size, action_size, num_agents)

    scores = run(agent, args.algorithm, env, num_episodes=10000)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(start=1, stop=len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    fig.savefig(f'results/{args.algorithm}/result.png')
    plt.close(fig)

