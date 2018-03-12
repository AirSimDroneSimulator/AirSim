from AirSimClient import *
from PIL import Image

import numpy as np
import tensorflow as tf
import os

class ReplayMemory:
	# memory for experience replay
	def __init__(self, size, shape, history_length = 4):
		self._count = 0 # length of current memory
		self._pos = 0 # number of experiences so far
		self._memory_size = size
		self._dim = shape
		self._history_length = history_length
		self._states = np.zeros((size,) + shape, dtype=np.float32)
		self._actions = np.zeros(size, dtype=np.uint8)
		self._rewards = np.zeros(size, dtype=np.float32)
		self._terminals = np.zeros(size, dtype=np.float32)

	def __len__(self):
		return self._count
	
	def append(self, state, action, reward, terminal):
		assert state.shape == self._dim
		self._states[self._pos] = state
		self._actions[self._pos] = action
		self._rewards[self._pos] = reward
		self._terminals[self._pos] = terminal
		
		self._count = max(self._count, self._pos + 1)
		self._pos = (self._pos + 1) % self._memory_size
	
	def sample(self, size):
		# return indices of sampled experiences
		count, pos, terminals = self._count-1, self._pos, self._terminals
		history_len = self._history_length
		indices = []
		
		while len(indices) < size:
			index = np.random.randint(history_len, count)
			if index not in indices:
				if not (index >= pos > index - history_len):
					if not terminals[(index - history_len):index].any():
						indices.append(index)
		return indices
	
	def minibatch(self, size):
		# Generate a minibatch with the number of samples specified by the size parameter
		indices = self.sample(size)
		pre_states = np.array([self.get_state(index) for index in indices], dtype=np.float32)
		post_states = np.array([self.get_state(index + 1) for index in indices], dtype=np.float32)
		actions = self._actions[indices]
		rewards = self._rewards[indices]
		terminals = self._terminals[indices]
		return pre_states, actions, post_states, rewards, terminals

	def get_state(self, index):
		# return state with specified index
		# a state is of shape (history_length, input_shape)
		if self._count == 0:
			raise("Empty memory!")
		index %= self._count
		history_length = self._history_length
		if index >= history_length:
			return self._states[(index - (history_length - 1)):index + 1, ...]
		else:
			indexes = np.arange(index - history_length + 1, index + 1)
			return self._states.take(indexes, mode='wrap', axis=0)
	
class History:
	# Accumulator keeping track of the N previous frames to be used by the agent
	def __init__(self, shape):
		self._buffer = np.zeros(shape, dtype=np.float32)
	
	def append(self, state):
        # Append state to the history
		self._buffer[:-1] = self._buffer[1:]
		self._buffer[-1] = state
		
	def get(self):
		return self._buffer
	
	def reset(self):
		self._buffer.fill(0)

def loss_function(y, y_hat):
		L = tf.reduce_sum((y - y_hat) ** 2)
		return L
		
class DQN_agent:
	# Deep Q network agent
	def __init__(self, input_shape, action_num, sess, 
				 learning_rate=0.00025, gamma=0.99, epsilon=0.1, train_after=50,
				 minibatch_size = 32, memory_size=50000, target_update_interval=1000, train_interval=4):
		self.input_shape = input_shape
		self.action_num = action_num
		self.gamma = gamma
		self.minibatch_size = minibatch_size
		self.epsilon = epsilon
		self.train_interval = train_interval
		self.train_after = train_after
		self.sess = sess
		self.target_update_interval = target_update_interval
		
		self.replay_memory = ReplayMemory(memory_size, input_shape[1:])
		self.history = History(input_shape)
		self.num_action_taken = 0
		self.training = False
		
		self.X_Q = tf.placeholder(tf.float32, [None] + list(input_shape))
		self.X_t = tf.placeholder(tf.float32, [None] + list(input_shape))
		self.Q_network = self.build_network("Q_network", self.X_Q)
		self.target_network = self.build_network("target_network", self.X_t)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		
		with tf.variable_scope("optimizer"):
			self.actions = tf.placeholder(tf.int32, [None], name="actions")
			self.rewards = tf.placeholder(tf.float32, [None], name="rewards")
			self.terminals = tf.placeholder(tf.float32, [None], name="terminals")
			actions_one_hot = tf.one_hot(self.actions, self.action_num)
			pred_Q = tf.reduce_sum(self.Q_network * actions_one_hot, axis=1)
			Q_target = self.rewards + self.gamma * tf.reduce_max(self.target_network, axis=1) * (1-self.terminals)
			self.loss = loss_function(Q_target, pred_Q)
			self.train_step = self.optimizer.minimize(self.loss)
		
	def build_network(self, scope_name, X):
		with tf.variable_scope(scope_name):
		
			conv1 = tf.layers.conv2d(inputs=X, filters=16, kernel_size=[8,8], strides=[4,4], padding='same', activation=tf.nn.relu)
			conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[4,4], strides=[2,2], padding='same', activation=tf.nn.relu)
			conv3 = tf.layers.conv2d(inputs=conv2, filters=32, kernel_size=[3,3], strides=[1,1], padding='same', activation=tf.nn.relu)
			flatten = tf.contrib.layers.flatten(conv3)
			fc1 = tf.layers.dense(inputs=flatten, units=256, activation=tf.nn.relu)
			fc2 = tf.layers.dense(inputs=fc1, units=self.action_num)
		return fc2
	
	def update_target_net(self, dest_scope="target_network", src_scope="Q_network"):
		# return tf operations that copy weights to target network
		ops = []
		src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope)
		dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope)
		
		for src_var, dest_var in zip(src_vars, dest_vars):
			ops.append(dest_var.assign(src_var.value()))
		
		return ops
	
	def act(self, state):
		# select action follow epsilon greedy
		self.history.append(state)
		rand = np.random.random()
		
		if rand > self.epsilon:
			env_history = self.history.get()
			env_history = np.reshape(env_history, [1]+list(self.input_shape))
			Q_values = self.sess.run(self.Q_network, feed_dict={self.X_Q : env_history})
			action = np.argmax(Q_values)
		else:
			action = np.random.randint(low=0, high=self.action_num)
		self.num_action_taken += 1
		return action
	
	def observe(self, old_state, action, reward, terminal):
		# observe environment after choosing action by act()
		# old_state : old state before taking the action
		# action : action selected by policy
		# reward : reward for applying action on old state
		# terminal : whether the action terminates an episode
		if terminal:
			self.history.reset()
		self.replay_memory.append(old_state, action, reward, terminal)
	
	def train(self):
		# train the neural network after a few number of actions
		# so that there are enough training samples in replay memory
		if self.num_action_taken >= self.train_after:
			# update target net every __ interval
			if (self.num_action_taken % self.target_update_interval == 0):
				self.sess.run(self.update_target_net())
			
			if (self.num_action_taken % self.train_interval) == 0:
				pre_states, actions, post_states, rewards, terminals = self.replay_memory.minibatch(self.minibatch_size)
				
				sess.run(self.train_step, feed_dict={self.X_Q:pre_states, 
													 self.X_t:post_states, 
													 self.actions:actions, 
													 self.rewards:rewards, 
													 self.terminals:terminals})

	def save_model(self, saver, model_path):
		# save model for future use
		saver.save(self.sess, model_path)
		print("Model saved in %s!" % model_path)
		
	def restore_model(self, saver, model_path):
		# restore previous model
		saver.restore(self.sess, model_path)
		print("Model restored!")
		
				
def transform_input(responses):
	# return a numpy array representation of image
    img1d = np.array(responses[0].image_data_float, dtype=np.float)
    img1d = 255/np.maximum(np.ones(img1d.size), img1d)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
    image = Image.fromarray(img2d)
    im_final = np.array(image.resize((84, 84)).convert('L')) 

    return im_final

def interpret_action(action):
    scaling_factor = 0.25
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
    return quad_offset
	
def compute_reward(destination, quad_state, quad_vel, collision_info):
	initial_pos = np.array((0, 0, -1.5))
	destination = np.array(destination)
	thresh_dist = 7
	beta = 0.5
	quad_pos = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
	
	if collision_info.has_collided:
		reward = -100
	else:
		initial_dist = np.linalg.norm(destination-initial_pos)
		dist = np.linalg.norm(quad_pos - destination)
		if dist > initial_dist + thresh_dist:
			reward = -10
		else:
			reward_dist = math.exp(-beta * dist)
			reward_speed = np.linalg.norm([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val]) - 0.5
			reward = reward_dist + reward_speed
	
	return reward
	
def is_terminal(reward):
	return reward <= -10
	
if __name__ == "__main__":
	input_shape = (4, 84, 84)
	action_num = 7
	epoch, max_epoch = 0, 10
	destination = (32, 38, -4)
	
	# connect to the AirSim simulator 
	client = MultirotorClient()
	client.confirmConnection()
	client.enableApiControl(True)
	client.armDisarm(True)
	
	client.moveToPosition(0,0,-5,5)
	time.sleep(0.5)
	
	with tf.device("/cpu:0"):
	
		with tf.Session() as sess:
		
			agent = DQN_agent(input_shape, action_num, sess)
			saver = tf.train.Saver()
			
			# check whether there exists pre-trained model
			# if yes, keep on training
			dirname = os.path.dirname(os.path.abspath(__file__))
			model_path = os.path.join(dirname, "dqn.ckpt")
			if os.path.exists(os.path.join(dirname, "checkpoint")):
				agent.restore_model(saver, model_path)
				agent.training = True
			else:
				sess.run(tf.global_variables_initializer())
			
			responses = client.simGetImages([ImageRequest(3, AirSimImageType.DepthPerspective, True, False)])
			current_state = transform_input(responses)
			
			while epoch < max_epoch:
				action = agent.act(current_state)
				quad_offset = interpret_action(action)
				quad_vel = client.getVelocity()
				client.moveByVelocity(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1], quad_vel.z_val+quad_offset[2], 5)
				time.sleep(0.5)
				
				quad_state = client.getPosition()
				quad_vel = client.getVelocity()
				collision_info = client.getCollisionInfo()
				reward = compute_reward(destination, quad_state, quad_vel, collision_info)
				terminal = is_terminal(reward)
				
				agent.observe(current_state, action, reward, terminal)
				agent.train()
				
				if terminal:
					print("epoch %d finished" % (epoch))
					client.reset()
					client.enableApiControl(True)
					client.armDisarm(True)
					client.moveToPosition(0,0,-5,5)
					time.sleep(0.5)
					epoch += 1
					
				responses = client.simGetImages([ImageRequest(3, AirSimImageType.DepthPerspective, True, False)])
				current_state = transform_input(responses)
			
			agent.save_model(saver, model_path)