
Object Detector through an UAV in unreal Airsim environment
=====
Hypothesis
------
By definition:
>An unmanned aerial vehicle (UAV), commonly known as a drone, is an aircraft without a human pilot aboard. 

Nowadays, UAV can be very small and contribute in many circumstances. In this project, we consider about one of them that using a drone with cameras to search for specific object in unreal world. 
Simply, it can be separated into two problems. The first one is search for a path in 3D environment to avoid collision. Another is searching for a specific object through the drone cameras and planning for its route in searching area.

Questions
---------
###Questions in 3D path finding
How do we define the environment or to say what features are required?
How to avoid collision through the features provided?
###Questions in Object Detection and route plan
How to mark an object in a sense image?
How to plan an efficient route for a certain area?

Methods and Plans
-------
All our work will based on Microsoft AirSim simulator.
>[AirSim](https://github.com/Microsoft/AirSim) is a simulator for drones, cars and more built on Unreal Engine. It is open-source, cross platform and supports hardware-in-loop with popular flight controllers such as PX4 for physically and visually realistic simulations. It is developed as an Unreal plugin that can simply be dropped in to any Unreal environment you want.
>Our goal is to develop AirSim as a platform for AI research to experiment with deep learning, computer vision and reinforcement learning algorithms for autonomous vehicles. For this purpose, AirSim also exposes APIs to retrieve data and control vehicles in a platform independent way.

If it's possible, we wold try it in the real world through a real drone at last.

###3D path finding
We decide to use deep reinforcement learning algorithms and some basic decision trees to solve this problem. Simply we plan to divided the whole project into three steps:
####1. 3D gridworld environment
We can consider the whole environment as a 3D gridworld and each time the drone is able to move through one grid. In this environment, only position and velocity of the drone are provided. Here we want to check the availability of deep reinforcement learning algorithms.
####2. height control environment
In the second step, we apply image features from the drone cameras in order to avoid collision. The action in this step is the z axis velocity at a time. It should be continuous. Here we want to test if we can solve the problem with image features.
####3. action control environment
In the third step, actions on all axes are open for the agent, we want to design a policy which can move from A to B.

###Object Detection
We will create some new objects, like some Hello Kitty, in the unreal world. Then design some algorithms for detecting and route planning in a certain area.



