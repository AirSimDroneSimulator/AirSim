import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from buffer import MemoryBuffer
import model
import model_P
import copy
import utils

init_settings = {}
init_settings["learning_rate"] = 0.001
init_settings["reward_decay"] = 0.9
init_settings["buffer_length"] = 50000
init_settings["batch_size"] = 32
init_settings["Actor"] = None
init_settings["Critic"] = None
init_settings["TAU"] = 0.001
init_settings["state_dim"] = 6
init_settings["action_dim"] = 1
init_settings["action_lim"] = 1


USE_CUDA = torch.cuda.is_available()

class Variable(torch.autograd.Variable):
	def __init__(self, data, *args, **kwargs):
		if USE_CUDA:
			data = data.cuda()
			super(Variable, self).__init__(data, *args, **kwargs)

class DDPG():

	def __init__(self,settings):
		
		self.settings = init_settings.copy()
		self.settings.update(settings)
		
		self.state_dim = self.settings["state_dim"]
		self.action_dim = self.settings["action_dim"]
		self.action_lim = self.settings["action_lim"]
		
		self.noise = utils.OrnsteinUhlenbeckActionNoise(self.settings["action_dim"])
		self.buffer = MemoryBuffer(self.settings["buffer_length"])
		self.batch_size = self.settings["batch_size"]
		self.gamma = self.settings["reward_decay"]
		self.tau = self.settings["TAU"]
		self.lr = self.settings["learning_rate"]
		
		self.actor = self.settings["Actor"]
		self.critic = self.settings["Critic"]
		self.target_actor = copy.deepcopy(self.actor)
		self.target_critic = copy.deepcopy(self.critic)
		
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),self.lr)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),self.lr)

		utils.hard_update(self.target_actor, self.actor)
		utils.hard_update(self.target_critic, self.critic)
		
	def choose_action(self, state):
		action = self.actor.forward(Variable(torch.from_numpy(np.array([state])).float())).detach()
		#action = self.actor.forward(state).detach()
		new_action = action.data.numpy() + (self.noise.sample() * self.action_lim) 
		return new_action[0]

		
	def choose_action_test(self, state):
		action = self.target_actor.forward(Variable(torch.from_numpy(np.array([state])).float())).detach()
		#action = self.target_actor.forward(state).detach()
		return action.data.numpy()[0]
		
	def learn(self, times=1):
		for i in range(times):
			if self.buffer.len < self.batch_size:
				return
			s, a, r, s_ = self.buffer.sample(self.batch_size)
			s = Variable(torch.from_numpy(s).float())
			a = Variable(torch.from_numpy(a).float())
			r = Variable(torch.from_numpy(r).float())
			s_ = Variable(torch.from_numpy(s_).float())
			
			#print (s)
			
			# for Critic
			#a_ = self.target_actor.forward(s_).detach().data.numpy()
			a_ = self.target_actor.forward(s_).detach()
			next_Q = self.target_critic.forward(s_, a_).detach()
			y = r + self.gamma * next_Q
			y_pred = self.critic.forward(s,a)
			
			loss_critic = F.smooth_l1_loss(y_pred, y)
			self.critic_optimizer.zero_grad()
			loss_critic.backward()
			self.critic_optimizer.step()
			
			# for Actor
			#pred_a = self.actor.forward(s).data.numpy()
			pred_a = self.actor.forward(s)
			loss_actor = -1*torch.sum(self.critic.forward(s, pred_a))
			self.actor_optimizer.zero_grad()
			loss_actor.backward()
			self.actor_optimizer.step()

			utils.soft_update(self.target_actor, self.actor, self.tau)
			utils.soft_update(self.target_critic, self.critic, self.tau)
		
	def add_data(self,s,a,r,s_):
		self.buffer.add(s,a,r,s_)
		
		
def save(actor, critic, name = "test",):
	torch.save(actor, name + "_actor.pt")
	torch.save(critic, name + "_critic.pt")
	print ("models saved")
			
def load(name):
	actor = torch.load(name + "_actor.pt")
	critic = torch.load(name + "_critic.pt")
	print ('models loaded')
	return actor, critic
		
		
		
		
		
		
		
		

