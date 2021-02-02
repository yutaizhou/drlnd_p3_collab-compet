Project 3 of the Udacity Deep Reinforcement Learning Nanodegree Program

**Project Details**

The envionrment is Tennis from Unity ML Agents. It is a multi-agent environment where two agents, each controlling a tennis racket from opposite sides of a tennis net, must keep the ball bouncing over the net and in play.

State Space: Continuous. 8 element vector that corresponds to the position and the velocity of the ball and racket. The environment does further preprocessing for stacking observation from three frames into one vector, thus forming a 3x8=24 element vector. Each agent gets its own local observation. 

Action Space: Contiuous. 2 element vector that corresponds horizontal movement relative to the net, and jumping.

Reward: An agent gains +0.1 reward if it hits the ball over the net, but loses -0.01 if it lets a ball reach the ground or hit the ball out of bounds. 

Episodic task, where each episode is 1000 steps through the environment. 

The task is considered solved when the meta-agent controlling both players gets an average score of +0.5 over 100 episodes. The score of an episode for a meta agent is taken as the maximum over the scores individually obtained by the two agents it controls.

Reward Structure:
In this kind of multi-agent environments, the kind of reward signals provided by the environment can dictate whether the agents' optimal behavior that maximizes the rewards is cooperative, competitive, or mixed. In this case, where +0.1 is awarded for hitting the ball over a net, and only -0.01 is dealt as punishment for letting the ball hit the ground or go out of bounds, the agents will have much higher incentive to learn to keep the ball volleying back and forth.


**Repository Layout**
- results/: the latest run from the implemented algorithms, containing the score numpy file, plot of score, agent model, and progress log. 
- src/: source code for the agent, which is separated into code specific to agent itself, reacher environment, and utilities
- Continuous_Control.ipynb: this is simply a file copied over from the official repository. Safe to ignore.
- run.py: code run running, evaluating, and logging the agent. 

**Getting Started** 

Create a conda environment from the environment.yml file:
conda env create -f environment.yml

Activate the newly created environment:
conda activate drlnd

**Instructions**

At the time of writing (Jan 31, 2021), 1 algorithm has been implemented: DDPG.

To run DDPG without loading any weights, run the following:
```
python run.py DDPG
```
Note: unless the directory path has been changed in run.py, this will overwrite the previous results.

