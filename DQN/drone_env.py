import AirSimClient
import time
import numpy as np

goal_threshold = 10
np.set_printoptions(precision=3, suppress=True)

class drone_env:
    def __init__(self,start = [0,0,-5],aim = [32,38,-4]):
        self.start = start
        self.aim = aim
        self.client = AirSimClient.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.threshold = goal_threshold
        
    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.moveToPosition(self.start[0],self.start[1],self.start[2],5,max_wait_seconds = 10)
        time.sleep(2)
        
        
    def isDone(self):
        pos = self.client.getPosition()
        if distance(self.aim,pos) < self.threshold:
            return True
        return False
        
    def moveByDist(self,diff):
        self.client.moveByVelocity(diff[0], diff[1], diff[2], 1)
        time.sleep(1)
        
        return 0
        
    def render(self,extra1 = "",extra2 = ""):
        pos = v2t(self.client.getPosition())
        goal = distance(self.aim,pos)
        print (extra1,"distance:",int(goal),"position:",pos.astype("int"),extra2)
        
    def help(self):
        print ("drone simulation environment")
        
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