3D Path Finding
=======

## Requirement


tensorflow 1.8
AirSim 1.1.8


## OpenAI like env

EnvGridWorld:

discrete environment, position and velocity as state, type of directions as actions.

EnvImgConrtol:

discrete environment, images as input, keep height or fly higher as actions. Aim for maintain UAV's height.

EnvHeightControl:

contiuous environment, images and height as input, difference of height as action. Aim for maintain UAV's height.


## Demo

DDPG

![Image text](https://github.com/AirSimDroneSimulator/AirSim/blob/master/3D_path_finding/demo/DDPG.gif)

DQN

![Image text](https://github.com/AirSimDroneSimulator/AirSim/blob/master/3D_path_finding/demo/DQN.gif)
