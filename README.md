# Multi Agent RL Block Pushing Environment with Soft Actor Critic Agents
## Environment
- The task is about two agents pushing a rectangular block to a target position
- Each agent is located at the ends of the block, and can only exert a forward or backward velocity i.e. one dimensional continuous action space
- The objective is to get the two agents to cooperate and achieve a shared task, without observing each other's policy
  - Green Rectangle: Goal
  - Red Rectangle: Block
  - Yellow Dots: agents pushing the Block
<p align="center">
<img width="625" height="450" src="https://user-images.githubusercontent.com/79006977/172347780-7b960569-0813-4ac0-bae2-6a284bb551e1.png">
</p>

## Reward
 - Shared reward, which scales with euclidean distance from goal
 - Large reward upon reaching goal
 - small cost for every step in the environment


## RL
- A basic Soft Actor Critic was implemented, adapted from [Phil Atbor](https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/SAC)
## Requirements
- torch
- numpy
- pymunk
- pygame
- keyboard
## Run
- create a local 'plots' and 'tmp/sac' folder
- run the main_sac file
