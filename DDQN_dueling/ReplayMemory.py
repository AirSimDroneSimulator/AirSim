from collections import deque

import random

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
        actions = [data[1] for data in minibatch]
        rewards = [data[2] for data in minibatch]
        next_states = [data[3] for data in minibatch]
        terminals = [data[4] for data in minibatch]
        return states, actions, rewards, next_states, terminals