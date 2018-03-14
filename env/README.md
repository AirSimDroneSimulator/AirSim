gymlike game environment
=======

Introduction
--------
gym like test framework.
currently only have one game: drone_env_gridworld
which moves from A to B through only positions and velocity

state: list [pos_x,pos_y,pos_z,vel_x,vel_y,vel_z]
pos -> Position
vel -> Velocity

action_dim = 1
action = 0 quad_offset = (0, 0, 0)
action = 1 quad_offset = (scaling_factor, 0, 0)
action = 2 quad_offset = (0, scaling_factor, 0)
action = 3 quad_offset = (0, 0, scaling_factor)
action = 4 quad_offset = (-scaling_factor, 0, 0)
action = 5 quad_offset = (0, -scaling_factor, 0)
action = 6 quad_offset = (0, 0, -scaling_factor)

env.reset():
	reset the environment, and move to the start position
	it returns the state

env.step(action)
	returns state_,reward,done,info

env.render(extra1,extra2)
	print some information,with some extra strings