from agent import DQN_agent
from drone_env import drone_env_gridworld

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

EPISODE = 500

def main():

    with tf.device("/gpu:0"):
	
        config = tf.ConfigProto(allow_soft_placement = True)
        with tf.Session(config=config) as sess:
        
            env = drone_env_gridworld()
            state = env.reset()
            input_shape = state.shape[0]
            action_num = 7
            agent = DQN_agent(sess, input_shape, action_num)
            sess.run(tf.global_variables_initializer())
            
            reward_list = []
            e, success, episode_reward = 0, 0, 0
            
            while e < EPISODE:
            
                action = agent.act(state)
                next_state, reward, terminal, info = env.step(action)
                episode_reward += reward
                agent.observe(state, action, reward, next_state, terminal)
                agent.train()
                state = next_state
                
                if terminal:
                    
                    state = env.reset()
                    reward_list.append(episode_reward)
                    if info == "success":
                        success += 1
                    print("episode {} finish, reward: {}, total steps: {}, total success: {}".format(e, episode_reward, agent.num_action_taken, success))
                    episode_reward = 0
                    e += 1
    
    plt.plot(reward_list)
    plt.show()

if __name__ == "__main__":
    main()