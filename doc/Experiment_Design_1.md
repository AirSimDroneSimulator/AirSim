Experiment Design
===============

Title
-------

The effect of deep reinforcement learning algorithms on the performance of agent for 3D path finding problem.

HYPOTHESIS
--------- 

If we apply the deep reinforcement learning algorithms other than some naive approaches, then it provide a better performance when environment changes in this 3D path finding problems. 

Independent Variables
-----------

Some DRL algorithms

Levels of Indenpent variable and numbers of repeated trails
---------

Level 1: Discretize the environment and use only position and velocity as states.

Level 2: Using images as input to avoid collision. Continuous action space in height control.

Level 3: Continuous action space in all dimensions and apply all features as states.

Dependent variable and how measure
--------

Measurement:

Compare the success rates, time costs and especially the extra workload (how much the human knowledge involved).

Constants
-----------

1. Using Neighborhood Airsim drone Environment.

2.  Position, velocity and depth view image from drone camera from python APIs as input features.
 