example:
import drone_env

env = drone_env.drone_env_gridworld(start = [0,0,-5],aim = [32,38,-4],scaling_factor = 5)
	the gridworld game of the UAV, goal is to move from start to aim. each step will move a certain distance on a certain vector

gridworld of the UAV:
	state: list [pos_x,pos_y,pos_z,vel_x,vel_y,vel_z]
	pos -> Position
	vel -> Velocity
	step: an int in the list [0,1,2,3,4,5,6]

        scaling_factor = self.scaling_factor
        if action == 0:
            quad_offset = (0, 0, 0)
        elif action == 1:
            quad_offset = (scaling_factor, 0, 0)
        elif action == 2:
            quad_offset = (0, scaling_factor, 0)
        elif action == 3:
            quad_offset = (0, 0, scaling_factor)
        elif action == 4:
            quad_offset = (-scaling_factor, 0, 0)    
        elif action == 5:
            quad_offset = (0, -scaling_factor, 0)
        elif action == 6:
            quad_offset = (0, 0, -scaling_factor)
        

env.reset():
	reset the environment, and move to the start position
	it returns the state

env.step(action)
	returns state_,reward,done,info

env.render(extra1,extra2)
	print some information,with some extra strings