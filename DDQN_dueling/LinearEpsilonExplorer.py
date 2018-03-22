import numpy as np

class LinearEpsilonExplorer:
    # instance for linear epsilon greedy policy
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        
        self.step_size = (start-end) / steps
    
    def choose_random_action(self, action_num):
        return np.random.randint(low=0, high=action_num)
    
    def explore(self, num_action_taken):
        return np.random.random() <= self._epsilon(num_action_taken)
        
    def _epsilon(self, num_action_taken):
        if num_action_taken >= self.steps:
            return self.end
        else:
            return self.start - num_action_taken * self.step_size