import tensorflow as tf
import numpy as np
import os
from DDPG import DDPG_agent
from drone_env import drone_env_heightcontrol

PATH = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(PATH, "test")
tf.set_random_seed(22)

def main():

	with tf.device("/gpu:0"):

		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		with tf.Session(config=config) as sess:

			globe_episode = tf.Variable(0, dtype=tf.int32, trainable=False, name='globe_episode')
			env = drone_env_heightcontrol()
			state_shape = 1
			action_bound = 1
			action_dim = 1
			agent = DDPG_agent(sess, state_shape, action_bound, action_dim)
			saver = tf.train.Saver()

			if not agent.load(saver, DIR):
				sess.run(tf.global_variables_initializer())
				if not os.path.exists(DIR):
					os.mkdir(DIR)
			else:
				print ("model loaded-------------------------")

			e, success, episode_reward = 0, 0, 0
			state = env.reset()

			while True:

				action = agent.act(state,noise = False)
				next_state, reward, terminal, info = env.step(action)
				#print(reward)
				episode_reward += reward
				state = next_state

				if terminal:

					
					if info == "success":
						success += 1
					print("episode {} finish, reward: {}, total success: {}".format(e, episode_reward, success))
					episode_reward = 0
					e += 1
					total_episode = sess.run(globe_episode.assign_add(1))
					state = env.reset()


if __name__ == "__main__":
	main()