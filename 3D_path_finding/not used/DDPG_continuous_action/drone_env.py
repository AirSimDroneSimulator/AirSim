import AirSimClient
import time
import numpy as np
from PIL import Image
import cv2

goal_threshold = 10
np.set_printoptions(precision=3, suppress=True)
IMG_SIZE = (56,56)


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
		
	def moveByDist(self,diff, forward = False, duration = 1,yaw = None):
		if yaw is None:
			yaw = AirSimClient.YawMode()
			yaw.is_rate = not forward
		self.client.moveByVelocity(diff[0], diff[1], diff[2], duration ,drivetrain = AirSimClient.DrivetrainType.ForwardOnly, yaw_mode = yaw)
		time.sleep(1)
		
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
	def __init__(self,start = [-23,0,-10],aim = [[-23,125,-10],[-23,0,-10]],scaling_factor = 2,img_size = [56,56],view = False):
		drone_env.__init__(self,start,aim)
		self.scaling_factor = scaling_factor
		self.current_goal = -1
		self.threshold = self.threshold//2
		self.height = self.aim[self.current_goal][2]
		self.view = view
		self.height_limit = abs(self.height)
		
	def reset(self):
		cv2.destroyAllWindows()
		self.current_goal = -1
		
		yaw = AirSimClient.YawMode()
		yaw.is_rate = False
		yaw.yaw_or_rate = 
		drone_env.reset(self)
		time.sleep(2)
		self.change_goal()
		state = self.getState()
		self.state = state
		return state
		
	def getState(self):
		pos = v2t(self.client.getPosition())
		vel = v2t(self.client.getVelocity())
		img = self.getImg()
		#state = np.array([img, np.append(pos, vel)])
		state = np.array([img, np.array(pos[2]-self.height).astype("float32")])
		
		return state
		
	def step(self,action):
		pos = v2t(self.client.getPosition())
		dpos = self.aim[self.current_goal] - pos
		
		action = action[0]
		if abs(action) > 1:
			#print ("action value error", action)
			action = action / abs(action)
		
		temp = np.sqrt(dpos[0]**2 + dpos[1]**2)
		#print (self.client.getPosition())
		dx = dpos[0] / temp * self.scaling_factor
		dy = dpos[1] / temp * self.scaling_factor
		dz = - action * self.scaling_factor
		#print (dx,dy,dz)
		drone_env.moveByDist(self,[dx,dy,dz],forward = True,duration = 10)
		
		state_ = self.getState()
		pos = state_[1]
		
		info = None
		done = False
		reward = self.rewardf(self.state,state_)
		
		if self.isDone():
			self.change_goal()
			info = "chang goal"
		if self.client.getCollisionInfo().has_collided:
			reward = -100
			done = True
			info = "collision"
		if abs(dpos[2]) > self.height_limit:
			reward = -100
			done = True
			info = "too high"
			
		self.state = state_
		
		return state_,reward,done,info
		
	def isDone(self):
		pos = v2t(self.client.getPosition())
		aim = self.aim[self.current_goal]
		pos[2] = aim[2]
		if distance(aim,pos) < self.threshold:
			return True
		return False
	
	def change_goal(self):
		self.current_goal += 1
		if self.current_goal == len(self.aim):
			self.current_goal = 0
		pos = v2t(self.client.getPosition())
		dif = self.aim[self.current_goal] - pos
		ang = np.arctan2(dif[1],dif[0]) * 180 / np.pi
		self.client.rotateToYaw(90,max_wait_seconds = 5)
		time.sleep(0.5)
		self.client.rotateToYaw(90,max_wait_seconds = 5)
		time.sleep(0.5)
		self.client.rotateToYaw(90,max_wait_seconds = 5)
		time.sleep(0.5)
		self.height = self.aim[self.current_goal][2]
		return 
		
	def rewardf(self,state,state_):
		pos = state[1]
		pos_ = state_[1]
		reward = - abs(pos_) + 5
		reward = reward * 20
		reward -= 0
		return reward
		
	def render(self,extra1 = "",extra2 = ""):
		pos = v2t(self.client.getPosition())
		h = pos[2]
		if self.view:
			self.img_view(self.state[0])
		print (extra1,"height:",int(h),"difference:",int(h-self.height),extra2)
		
	def getImg(self):
		'''
		responses = self.client.simGetImages([AirSimClient.ImageRequest(0, AirSimClient.AirSimImageType.DepthVis, True, False)])
		img1 = np.array(responses[0].image_data_float, dtype=np.float)
		img2 = img1.reshape(responses[0].height, responses[0].width, 1)[:,:,0]
		image = Image.fromarray(img2)
		im_final = np.array([np.array(image.resize(IMG_SIZE))] )
		return im_final
		'''
		responses = self.client.simGetImages([AirSimClient.ImageRequest(0, AirSimClient.AirSimImageType.DepthPerspective, True, False)])
		img1d = np.array(responses[0].image_data_float, dtype=np.float)
		img1d = 255/np.maximum(np.ones(img1d.size), img1d)
		img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

		image = Image.fromarray(img2d)
		im_final = np.array(image.resize(IMG_SIZE).convert('L')) 
		return im_final
		
	def img_view(self,img):
		cv2.imshow("view",img)
		key = cv2.waitKey(1) & 0xFF;
		return 
		
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
	dist = abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1]) + abs(pos1[2]-pos2[2]) 
		
	return dist