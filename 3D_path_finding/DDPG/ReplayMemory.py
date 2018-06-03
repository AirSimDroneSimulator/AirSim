from collections import deque
import pickle
import os
import random
import numpy as np

class ReplayMemory:
	def __init__(self, memory_size):
		self.buffer = deque()
		self.memory_size = memory_size
	
	def append(self, pre_state, action, reward, post_state, terminal):
		self.buffer.append((pre_state, action, reward, post_state, terminal))
		if len(self.buffer) >= self.memory_size:
			self.buffer.popleft()
	
	def sample(self, size):
		minibatch = random.sample(self.buffer, size)
		states = [data[0] for data in minibatch]
		actions = np.array([data[1] for data in minibatch])
		rewards = np.array([data[2] for data in minibatch])
		next_states = [data[3] for data in minibatch]
		terminals = np.array([data[4] for data in minibatch])
		return states, actions, rewards, next_states, terminals

	def save(self, dir):
		file = os.path.join(dir, 'replaymemory.pickle')
		with open(file, 'wb') as f:
			pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

	def load(self, dir):
		file = os.path.join(dir, 'replaymemory.pickle')
		with open(file, 'rb') as f:
			memory = pickle.load(f)
		return memory