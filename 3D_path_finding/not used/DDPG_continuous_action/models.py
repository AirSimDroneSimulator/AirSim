import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 0.003

'''
def transform(state,size):
	st = 0
	ed = 0
	res = []
	ed = st + np.array(size[0]).prod()
	temp = state[:,st:ed].contiguous().view([-1]+size[0])
	res.append(temp)
	st = ed
	ed = st + np.array(size[1]).prod()
	temp = state[:,st:ed].contiguous().view([-1]+[size[1]])
	if state.size()[0] == 1:
		temp = temp[0]
	res.append(temp)
	return res
'''

def transform(state,size):
	st = 0
	ed = 0
	res = []
	for si in size:
		ed = st + np.array(si).prod()
		temp = state[:,st:ed].contiguous().view([-1]+si)
		res.append(temp)
		st = ed
	return res
	
def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.FloatTensor(size).uniform_(-v, v)

class Critic(nn.Module):

	def __init__(self, state_dim, action_dim):

		super(Critic, self).__init__()
		
		self.val_state_dim = state_dim[1][0]
		self.img_state_dim = state_dim[0]
		self.size = state_dim
		self.action_dim = action_dim

		self.fcs1 = nn.Linear(self.val_state_dim,32)
		self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
		self.fcs2 = nn.Linear(32,16)
		self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

		self.conv1 = nn.Sequential(         # input shape (1, 56, 56)
			nn.Conv2d(1,4,5,1,2),          # output shape (32, 56, 56),kernel_size = 5, stride = 1, padding = 2
			nn.ReLU(),                      # activation
			nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (32, 28, 28)
		)		
		self.conv2 = nn.Sequential(         # input shape (32, 28, 28)
			nn.Conv2d(4,8,5,1,2),         # output shape (64, 28, 28)
			nn.ReLU(),                      # activation
			nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (64, 14, 14)
		)
		self.conv3 = nn.Sequential(         # input shape (64, 14, 14)
			nn.Conv2d(8,16,5,1,2),        # output shape (64, 14, 14)
			nn.ReLU(),                      # activation
			nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (128, 7, 7)
		)
		self.conv4 = nn.Sequential(     # input shape (128, 7, 7)
			nn.Conv2d(16,16,7,1,0),       # output shape (128, 1, 1)
			nn.ReLU(),                      # activation. 
		)
		
		
		self.fca1 = nn.Linear(action_dim,32)
		self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

		self.fc2 = nn.Linear(64,32)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(32,16)
		self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
		
		self.fc4 = nn.Linear(16,1)
		self.fc4.weight.data.uniform_(-EPS,EPS)

	def forward(self, state, action):

		res = transform(state,self.size)
		img = res[0]
		val = res[1]
		size = img.size(0)
		
		s1 = F.relu(self.fcs1(val))
		s2 = F.relu(self.fcs2(s1))   #16
		c1 = self.conv1(img)
		c2 = self.conv2(c1)
		c3 = self.conv3(c2)
		c4 = self.conv4(c3)
		c4 = torch.squeeze(c4) # 16
		c4 = c4.view(size,-1)
		sout = torch.cat((c4,s2),dim = 1) #32
		a1 = F.relu(self.fca1(action))
		#print (sout,a1)
		o1 = torch.cat((sout,a1),dim = 1)
		o2 = F.relu(self.fc2(o1))
		o3 = F.relu(self.fc3(o2))
		
		x = self.fc4(o3)

		return x


class Actor(nn.Module):

	def __init__(self, state_dim, action_dim, action_lim):

		super(Actor, self).__init__()


		self.val_state_dim = state_dim[1][0]
		self.img_state_dim = state_dim[0]
		self.size = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim

		self.fcs1 = nn.Linear(self.val_state_dim,32)
		self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
		self.fcs2 = nn.Linear(32,16)
		self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

		self.conv1 = nn.Sequential(         # input shape (1, 56, 56)
			nn.Conv2d(1,4,5,1,2),          # output shape (32, 56, 56),kernel_size = 5, stride = 1, padding = 2
			nn.ReLU(),                      # activation
			nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (32, 28, 28)
		)		
		self.conv2 = nn.Sequential(         # input shape (32, 28, 28)
			nn.Conv2d(4,8,5,1,2),         # output shape (64, 28, 28)
			nn.ReLU(),                      # activation
			nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (64, 14, 14)
		)
		self.conv3 = nn.Sequential(         # input shape (64, 14, 14)
			nn.Conv2d(8,16,5,1,2),        # output shape (64, 14, 14)
			nn.ReLU(),                      # activation
			nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (128, 7, 7)
		)
		self.conv4 = nn.Sequential(     # input shape (128, 7, 7)
			nn.Conv2d(16,16,7,1,0),       # output shape (128, 1, 1)
			nn.ReLU(),                      # activation. 
		)
		
		
		
		self.fc1 = nn.Linear(32,16)
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
		
		self.fc2 = nn.Linear(16,8)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(8,action_dim)
		self.fc3.weight.data.uniform_(-EPS,EPS)

	def forward(self, state):
		
		res = transform(state,self.size)
		img = res[0]
		val = res[1]
		size = img.size(0)
		
		s1 = F.relu(self.fcs1(val))
		s2 = F.relu(self.fcs2(s1))   #16
		c1 = self.conv1(img)
		c2 = self.conv2(c1)
		c3 = self.conv3(c2)
		c4 = self.conv4(c3)
		c4 =  torch.squeeze(c4)   #16
		c4 = c4.view(size,-1)
		sout = torch.cat((c4,s2),dim = 1) #32
		
		#print (sout)
		o1 = F.relu(self.fc1(sout))
		o2 = F.relu(self.fc2(o1))
		
		action = F.tanh(self.fc3(o2))

		action = action * self.action_lim

		return action



