import AirSimClient
import time
import copy
import numpy as np
from PIL import Image
import cv2

goal_threshold = 3
np.set_printoptions(precision=3, suppress=True)
IMAGE_VIEW = True

class drone_env:
	def __init__(self,start = [0,0,-5],aim = [32,38,-4]):
		self.start = np.array(start)
		self.aim = np.array(aim)
		self.client = AirSimClient.MultirotorClient()
		self.client.confirmConnection()
		self.client.enableApiControl(True)
		self.client.armDisarm(True)
		self.threshold = goal_threshold
		
	def reset(self):
		self.client.reset()
		self.client.enableApiControl(True)
		self.client.armDisarm(True)
		self.client.moveToPosition(self.start.tolist()[0],self.start.tolist()[1],self.start.tolist()[2],5,max_wait_seconds = 10)
		time.sleep(2)
		
		
	def isDone(self):
		pos = self.client.getPosition()
		if distance(self.aim,pos) < self.threshold:
			return True
		return False
		
	def moveByDist(self,diff, forward = False):
		temp = AirSimClient.YawMode()
		temp.is_rate = not forward
		self.client.moveByVelocity(diff[0], diff[1], diff[2], 1 ,drivetrain = AirSimClient.DrivetrainType.ForwardOnly, yaw_mode = temp)
		time.sleep(0.5)
		
		return 0
		
	def render(self,extra1 = "",extra2 = ""):
		pos = v2t(self.client.getPosition())
		goal = distance(self.aim,pos)
		print (extra1,"distance:",int(goal),"position:",pos.astype("int"),extra2)
		
	def help(self):
		print ("drone simulation environment")
		
		
#-------------------------------------------------------
# grid world
		
class drone_env_gridworld(drone_env):
	def __init__(self,start = [0,0,-5],aim = [32,38,-4],scaling_factor = 5):
		drone_env.__init__(self,start,aim)
		self.scaling_factor = scaling_factor
		
	def interpret_action(self,action):
		scaling_factor = self.scaling_factor
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
		
		return np.array(quad_offset).astype("float64")
	
	def step(self,action):
		diff = self.interpret_action(action)
		drone_env.moveByDist(self,diff)
		
		pos_ = v2t(self.client.getPosition())
		vel_ = v2t(self.client.getVelocity())
		state_ = np.append(pos_, vel_)
		pos = self.state[0:3]
		
		info = None
		done = False
		reward = self.rewardf(self.state,state_)
		reawrd = reward / 50
		if action == 0:
			reward -= 10
		if self.isDone():
			done = True
			reward = 100
			info = "success"
		if self.client.getCollisionInfo().has_collided:
			reward = -100
			done = True
			info = "collision"
		if (distance(pos_,self.aim)>150):
			reward = -100
			done = True
			info = "out of range"
			
		self.state = state_
		
		return state_,reward,done,info
	
	def reset(self):
		drone_env.reset(self)
		pos = v2t(self.client.getPosition())
		vel = v2t(self.client.getVelocity())
		state = np.append(pos, vel)
		self.state = state
		return state
		
	def rewardf(self,state,state_):
		
		dis = distance(state[0:3],self.aim)
		dis_ = distance(state_[0:3],self.aim)
		reward = dis - dis_
		reward = reward * 1
		reward -= 1
		return reward
		
#-------------------------------------------------------
# height control
# continuous control
		
class drone_env_heightcontrol(drone_env):
	def __init__(self,start = [-23,0,-10],aim = [-23,125,-10],scaling_factor = 2,img_size = [64,64]):
		drone_env.__init__(self,start,aim)
		self.scaling_factor = scaling_factor
		self.aim = np.array(aim)
		self.height_limit = -30
		self.rand = False
		if aim == None:
			self.rand = True
			self.start = np.array([0,0,-10])
		else:
			self.aim_height = self.aim[2]
	
	def reset_aim(self):
		self.aim = (np.random.rand(3)*300).astype("int")-150
		self.aim[2] = -np.random.randint(10) - 5
		print ("Our aim is: {}".format(self.aim).ljust(80," "),end = '\r')
		self.aim_height = self.aim[2]
		
	def reset(self):
		if self.rand:
			self.reset_aim()
		drone_env.reset(self)
		self.state = self.getState()
		return self.state
		
	def getState(self):
		pos = v2t(self.client.getPosition())
		vel = v2t(self.client.getVelocity())
		img = self.getImg()
		state = [img, np.array([pos[2] - self.aim_height])]
		
		return state
		
	def step(self,action):
		pos = v2t(self.client.getPosition())
		dpos = self.aim - pos
		
		if abs(action) > 1:
			print ("action value error")
			action = action / abs(action)
		
		temp = np.sqrt(dpos[0]**2 + dpos[1]**2)
		dx = dpos[0] / temp * self.scaling_factor
		dy = dpos[1] / temp * self.scaling_factor
		dz = - action * self.scaling_factor
		#print (dx,dy,dz)
		drone_env.moveByDist(self,[dx,dy,dz],forward = True)
		
		state_ = self.getState()
		pos = state_[1][0]
		
		info = None
		done = False
		reward = self.rewardf(self.state,state_)
		
		if self.isDone():
			if self.rand:
				done = False
				#reward = 50
				#info = "success"
				self.reset_aim()
			else:
				done = True
				reward = 50
				info = "success"
			
		if self.client.getCollisionInfo().has_collided:
			reward = -50
			done = True
			info = "collision"
		if (pos + self.aim_height) < self.height_limit:
			done = True
			info = "too high"
			reward = -50
			
		self.state = state_
		reward /= 50
		norm_state = copy.deepcopy(state_)
		norm_state[1] = norm_state[1]/100
		
		return norm_state,reward,done,info
		
	def isDone(self):
		pos = v2t(self.client.getPosition())
		pos[2] = self.aim[2]
		if distance(self.aim,pos) < self.threshold:
			return True
		return False
		
	def rewardf(self,state,state_):
		pos = state[1][0]
		pos_ = state_[1][0]
		reward = - abs(pos_) + 5
		
		return reward
		
	def getImg(self):
		
		responses = self.client.simGetImages([AirSimClient.ImageRequest(0, AirSimClient.AirSimImageType.DepthPerspective, True, False)])
		img1d = np.array(responses[0].image_data_float, dtype=np.float)
		img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
		image = Image.fromarray(img2d)
		im_final = np.array(image.resize((64, 64)).convert('L'), dtype=np.float)/255
		im_final.resize((64,64,1))
		if IMAGE_VIEW:
			cv2.imshow("view",im_final)
			key = cv2.waitKey(1) & 0xFF;
		return im_final
		
def v2t(vect):
	if isinstance(vect,AirSimClient.Vector3r):
		res = np.array([vect.x_val, vect.y_val, vect.z_val])
	else:
		res = np.array(vect)
	return res

def distance(pos1,pos2):
	pos1 = v2t(pos1)
	pos2 = v2t(pos2)
	#dist = np.sqrt(abs(pos1[0]-pos2[0])**2 + abs(pos1[1]-pos2[1])**2 + abs(pos1[2]-pos2[2]) **2)
	dist = np.linalg.norm(pos1-pos2)
		
	return dist