from pushblock_env import Push_Block_Env, visualise, draw_box
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve
import pymunk,pygame
#from gym import wrappers

if __name__ == '__main__':
    env = Push_Block_Env(FPS=10)
    agent1 = Agent(input_dims=[8], n_actions=1,name='_agent1')
    agent2 = Agent(input_dims=[8], n_actions=1,name='_agent2')
    n_games = 50

    filename = 'pushblock.png'
  
    figure_file = 'plots/' + filename

    best_score = -2
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent1.load_models()
        agent2.load_models()
        #env.render(mode='human')

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        pygame.init()
        display = pygame.display.set_mode((env.display_size,env.display_size))
        clock = pygame.time.Clock()
        while not done:
            action1 = agent1.choose_action(observation)
            action2 = agent2.choose_action(observation)

            observation_, reward, done= env.step(action1,action2)
            display.fill((255,255,255))
            env.draw(display)
            draw_box(display,env.min_dist,env.max_dist)
            pygame.display.update()
            clock.tick(env.FPS*5)
            print(f'obs CG: {observation_[0:2]}, rew: {reward}, done: {done}')
            score += reward
            agent1.remember(observation, action1, reward, observation_, done)
            agent2.remember(observation, action2, reward, observation_, done)

            if not load_checkpoint:
                agent1.learn()
                agent2.learn()

            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent1.save_models()
                agent2.save_models()

        print('\n\n\nepisode ', i, 'score %.1f' % score, 'avg_score %.1f\n\n\n' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

    visualise(env,agent1, agent2) #visualise the trained agents
    pygame.quit()