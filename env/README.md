gymlike game environment
=======


gym like test framework.\

game 1: drone_env_gridworld\
------------------
which moves from A to B through only positions and velocity

drone_env_gridworld(self,start = [0,0,-5],aim = [32,38,-4],scaling_factor = 5)

state: list [pos_x,pos_y,pos_z,vel_x,vel_y,vel_z]\
pos -> Position\
vel -> Velocity\

action_dim = 1\
action = 0 quad_offset = (0, 0, 0)\
action = 1 quad_offset = (scaling_factor, 0, 0)\
action = 2 quad_offset = (0, scaling_factor, 0)\
action = 3 quad_offset = (0, 0, scaling_factor)\
action = 4 quad_offset = (-scaling_factor, 0, 0)\
action = 5 quad_offset = (0, -scaling_factor, 0)\
action = 6 quad_offset = (0, 0, -scaling_factor)\

env.reset():\
    reset the environment, and move to the start position\
    it returns the state

env.step(action)\
    returns state_,reward,done,info\

env.render(extra1,extra2)\
    print some information,with some extra strings\
    
game 2: drone_env_heightcontrol\
------------------
moves from A to B but only the height can be controlled\

drone_env_heightcontrol(self,start = [0,0,-5],aim = [32,38,-5],scaling_factor = 2,img_size = [224,224])

state: list [depth_img(224,224),position,velocity]\
action: float in the range -1 to 1 reflect to a distance moved at Z axis (positive means fly higher)\

similar methods as above:\
env.reset()\
env.step(action)\
env.render()\

env.getImg()\
get the depth image from the front camera (camera id 0)\

