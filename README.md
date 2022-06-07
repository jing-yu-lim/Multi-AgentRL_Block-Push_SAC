# Multi Agent RL Block Pushing Environment with Soft Actor Critic Agents
## Environment
- The task is about two agents pushing a rectangular block to a target position
- Each agent is located at the ends of the block, and can only exert a forward or backward acceleration i.e. one dimensional continuous action space
- ![image](https://user-images.githubusercontent.com/79006977/172347780-7b960569-0813-4ac0-bae2-6a284bb551e1.png "Rendering of the Pymunk physics using Pygame")
- Green Rectangle: Goal
- Red Rectangle: Block
- Yellow Dots: Agents pushing the Block

## RL
- A basic Soft Actor Critic was implemented, adapted from [Phil Atbor](https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/SAC)
# Run
- create a local 'plots' and 'tmp/sac' folder
- run the main_sac file
