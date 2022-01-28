from turtle import done
import pymunk, pygame
import keyboard
import random
import numpy as np

class Push_Block_Env:
    def __init__ (self,goal_dist=150, init_pos=400,FPS=50):
        self.goal_dist=goal_dist
        self.init_pos=init_pos
        self.max_dist = self.init_pos + self.goal_dist
        self.min_dist = self.init_pos - self.goal_dist 
        self.FPS=FPS
        self.space= pymunk.Space()  
        self.body = pymunk.Body()   #implicitly assumes the space to be 800x800
        self.body.position =init_pos,init_pos #starting position of the block
        self.body.angle=0
        self.shape = pymunk.Poly.create_box(self.body,(200,10)) #width:200, height:10
        self.shape.density=1
        self.space.add(self.body,self.shape)
        self.display_size=800

    def convert_coordinates(self,point):
        return point[0], self.display_size - point[1]

    def reset(self):
        self.body.position = 400,400
        self.body.angle = 0
        new_state=[]
        for v in self.shape.get_vertices():
            x,y = v.rotated(self.shape.body.angle) + self.shape.body.position #getting vertices in global frame
            new_state.append(x)
            new_state.append(y)
        return np.array(new_state)
    
    def step(self, action_1, action_2):
        #continuous action from 0 - 1
        full_force = 50000
        self.force((0,action_1*full_force),(-100,0)) #action_1 acting at the -100 (left) position)
        self.force((0,action_2*full_force),(100,0))  #action_2 acting at the  100 (right) position)
        self.space.step(1/self.FPS)                        #50 steps --> 1 second in pymunk space

        if self.body.position[1]>=self.max_dist:               #Goal: centre of gravity's y position reaching the top i.e. 800 pixel
            reward = 1
            done=True
        elif self.body.position[1] <=self.min_dist:              #-ve reward for reaching the bottom
            reward = -1
            done=True
        elif self.body.position[0] >=self.max_dist:            #-ve reward for reaching the right side
            reward = -1
            done=True
        elif self.body.position[0] <=self.min_dist:              #-ve reward for reaching the left side
            reward = -1
            done=True
        else:
            reward = - 0.01                        #smol -ve reward for every action in the episode
            done=False
        new_state=[]
        for v in self.shape.get_vertices():
            x,y = v.rotated(self.shape.body.angle) + self.shape.body.position #getting vertices in global frame
            new_state.append(x)
            new_state.append(y)
        return np.array(new_state), reward, done

    def force(self,f,point):
        self.body.apply_force_at_local_point(f,point)
    
    def status(self): #returns vertices for pygame to draw
        verts=[]
        for v in self.shape.get_vertices():
            x,y = v.rotated(self.shape.body.angle) + self.shape.body.position
            point = (int(x), int(y))
            verts.append(point)
        print(f'verts: {verts}')
        print(f'angle: {self.body.angle} | cg: {self.body.center_of_gravity}')
    
    def draw(self,display):
        verts=[]
        for v in self.shape.get_vertices():
            x,y = v.rotated(self.shape.body.angle) + self.shape.body.position
            point = (int(x), int(y))
            verts.append(point)
        print(f'verts: {verts}')
        print(f'angle: {self.body.angle} | cg: {self.body.center_of_gravity}')
        a = self.convert_coordinates(verts[0])
        b = self.convert_coordinates(verts[3])
        pygame.draw.line(display, (255,0,0), a,b,5) #draw line because pygame cannot rotate rectangles
#testing
def visualise(env, agent1=None,agent2=None):   
    random_pol=False
    if agent1==None or agent2==None:
        random_pol=True
    env=env
    pygame.init()
    FPS = 50
    display = pygame.display.set_mode((env.display_size,env.display_size))
    clock = pygame.time.Clock()
    done=False
    observation=env.reset()
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
        
        display.fill((255,255,255))
        if random_pol:
            a1= random.uniform(-1,1) #agent1's policy here
            a2= random.uniform(-1,1) #agent2's policy here
        else:
            a1=agent1.choose_action(observation)
            a2=agent2.choose_action(observation)

        new_obs,_,done=env.step(a1, a2)
        env.draw(display)
        draw_box(display,env.min_dist,env.max_dist)
        pygame.display.update()
        clock.tick(FPS)
        observation = new_obs

def draw_box(display,min_dist,max_dist):
    left_x, bot_y = min_dist,min_dist
    right_x, top_y = max_dist,max_dist
    pygame.draw.line(display, (255,255,0), (left_x,bot_y),(right_x,bot_y),2)
    pygame.draw.line(display, (255,255,0), (left_x,top_y),(right_x,top_y),2)
    pygame.draw.line(display, (255,255,0), (left_x,top_y),(left_x,bot_y),2)
    pygame.draw.line(display, (255,255,0), (right_x,top_y),(right_x,bot_y),2)



# visualise()
# pygame.quit()