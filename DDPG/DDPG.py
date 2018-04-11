import tensorflow as tf
import numpy as np
import os
from Actor import Actor
from Critic import Critic
from OUNoise import OrnsteinUhlenbeckActionNoise
from ReplayMemory import ReplayMemory

class DDPG_agent:
	def __init__(self, sess, state_shape, action_bound, action_dim,
				 memory_size=10000, minibatch_size=32, gamma=0.9, tau=0.001, train_after=50):
		self.actor = Actor(sess, action_bound, action_dim, state_shape, tau=tau)
		self.critic = Critic(sess, state_shape, action_dim, tau=tau)
		self.replay_memory = ReplayMemory(memory_size)
		self.sess = sess
		self.minibatch_size = minibatch_size
		self.action_bound = action_bound
		self.gamma = gamma
		self.train_after = train_after
		self.num_action_taken = 0
		self.action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(action_dim))
		
	def observe(self, state, action, reward, post_state, terminal):
		self.replay_memory.append(state, action, reward, post_state, terminal)
		
	def act(self, state):
		action = self.actor.act(state)
		noise = self.action_noise()
		action = np.clip(action+noise, -self.action_bound, self.action_bound)[0]
		self.num_action_taken += 1
		return action
	
	def update_target_nets(self):
		# update target net for both actor and critic
		self.sess.run([self.actor.update_ops, self.critic.update_ops])
	
	def train(self):
		if self.num_action_taken >= self.train_after:
			# 1 sample random minibatch from replay memory
			states, actions, rewards, post_states, terminals = \
				self.replay_memory.sample(self.minibatch_size)
			
			# 2 use actor's target net to select action for Si+1, denote as mu(S_i+1)
			mu_post_states = self.actor.target_action(post_states)
			
			# 3 use critic's target net to evaluate Q(S_i+1, a_i+1) and calculate td target
			Q_target = self.critic.target_net_eval(post_states, mu_post_states)
			rewards = rewards.reshape([self.minibatch_size, 1])
			terminals = terminals.reshape([self.minibatch_size, 1])
			td_target = rewards + self.gamma * Q_target * (1-terminals)
			
			# 4 update critic's online network
			self.critic.train(states, actions, td_target)
			
			# 5 predict action using actors online network and calculate the sampled gradients
			pred_actions = self.actor.predict_action(states)
			Q_gradients = self.critic.action_gradient(states, pred_actions)/self.minibatch_size
			
			# 6 update actor's online network
			self.actor.train(Q_gradients, states)
			
			# 7 apply soft replacement for both target networks
			self.update_target_nets()

	def save(self, saver, dir):
		path = os.path.join(dir, 'model')
		saver.save(self.sess, path)
		self.action_noise.save(dir)
		self.replay_memory.save(dir)

	def load(self, saver, dir):
		path = os.path.join(dir, 'checkpoint')
		ckpt = tf.train.get_checkpoint_state(os.path.dirname(path))

		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(self.sess, ckpt.model_checkpoint_path)
			self.action_noise = self.action_noise.load(dir)
			self.replay_memory = self.replay_memory.load(dir)
			self.train_after = 0
			return True
		return False