import numpy as np
import pickle
import os

class OrnsteinUhlenbeckActionNoise:
	# from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def save(self, dir):
        file = os.path.join(dir, 'ounoise.pickle')
        with open(file, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, dir):
        file = os.path.join(dir, 'ounoise.pickle')
        with open(file, 'rb') as f:
            noise = pickle.load(f)
        return noise