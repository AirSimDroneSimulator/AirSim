import tensorflow as tf
import numpy as np
import os
from DDPG import DDPG_agent
from drone_env import drone_env_heightcontrol

PATH = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(PATH, "data")
tf.set_random_seed(22)

def main():

	with tf.device("/gpu:0"):

		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.per_process_gpu_memory_fraction = 0.6
		with tf.Session(config=config) as sess:

			globe_episode = tf.Variable(0, dtype=tf.int32, trainable=False, name='globe_episode')
			env = drone_env_heightcontrol()
			state = env.reset()
			state_shape = 9
			action_bound = 1
			action_dim = 1
			agent = DDPG_agent(sess, state_shape, action_bound, action_dim)
			saver = tf.train.Saver()

			if not agent.load(saver, DIR):
				sess.run(tf.global_variables_initializer())
				if not os.path.exists(DIR):
					os.mkdir(DIR)

			e, success, episode_reward = 0, 0, 0

			while True:

				action = agent.act(state)
				next_state, reward, terminal, info = env.step(action)
				print(reward)
				episode_reward += reward
				agent.observe(state, action, reward, next_state, terminal)
				agent.train()
				state = next_state

				if terminal:

					state = env.reset()
					if info == "success":
						success += 1
					print("episode {} finish, reward: {}, total success: {}".format(e, episode_reward, success))
					episode_reward = 0
					e += 1
					total_episode = sess.run(globe_episode.assign_add(1))
					if e % 10 == 0:
						print("total training episode: {}".format(total_episode))
						agent.save(saver,DIR)

if __name__ == "__main__":
	main()