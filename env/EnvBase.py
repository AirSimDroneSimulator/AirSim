import env.AirSimClient as AirSimClient
import time
import copy
import numpy as np
from PIL import Image
import cv2
import tkinter as tk

np.set_printoptions(precision=3, suppress=True)

class EnvBase:
    def __init__(self):
        self.start = np.array([0, 0, -5])
        self.client = AirSimClient.MultirotorClient()
        self.client.confirmConnection()
        self.timestep = 0.5
        self.maxvel = 5

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.moveToPosition(self.start.tolist()[0], self.start.tolist()[1], self.start.tolist()[2], 5,
                                   max_wait_seconds=10)
        time.sleep(2)

    def reset_aim(self):
        self.aim = (np.random.rand(3) * 300).astype("int") - 150
        self.aim[2] = -np.random.randint(10) - 5
        print("Current aim is: {}".format(self.aim).ljust(80, " "))

    def moveByDist(self, diff, ori = None):
        diff = np.array(diff)
        if not ori:
            temp = AirSimClient.YawMode()
            temp.is_rate = False
            ori = temp
        self.client.moveByVelocity(diff.tolist()[0], diff.tolist()[1], diff.tolist()[2], self.timestep*2, drivetrain=AirSimClient.DrivetrainType.ForwardOnly,
                                   yaw_mode = ori)
        time.sleep(self.timestep)

    def moveByVel(self, diff, ori = None):
        vel = self.getVel()
        diff = np.clip(vel + diff,-self.maxvel,self.maxvel)
        if not ori:
            temp = AirSimClient.YawMode()
            temp.is_rate = False
            ori = temp
        self.client.moveByVelocity(diff.tolist()[0], diff.tolist()[1], diff.tolist()[2], self.timestep*2, drivetrain=AirSimClient.DrivetrainType.ForwardOnly,
                                   yaw_mode=ori)
        time.sleep(self.timestep)

    def arrive_aim(self,aim,threshold):
        pos = self.getPos()
        if distance(aim, pos) < threshold:
            return True
        return False

    def getPos(self):
        return v2t(self.client.getPosition())

    def getVel(self):
        return v2t(self.client.getVelocity())


def showImg(Img):
    cv2.imshow("view", Img)
    cv2.waitKey(1) & 0xFF


def distance(pos1, pos2):
    pos1 = v2t(pos1)
    pos2 = v2t(pos2)
    # dist = np.sqrt(abs(pos1[0]-pos2[0])**2 + abs(pos1[1]-pos2[1])**2 + abs(pos1[2]-pos2[2]) **2)
    dist = np.linalg.norm(pos1 - pos2)
    return dist


def v2t(vect):
    if isinstance(vect, AirSimClient.Vector3r):
        res = np.array([vect.x_val, vect.y_val, vect.z_val])
    else:
        res = np.array(vect)
    return res


if __name__ == "__main__":
    env = EnvBase()
    env.reset()
    print (env.getPos(),env.getVel())
    print (env.arrive_aim([0,0,-5],5))
    print (distance([0,0,-5],env.getPos()),"threshold:",5)
    env.moveByDist(np.array([2,0,-2]))
    env.moveByDist([2, 0, -2])
    env.moveByVel(np.array([0,-5,-3]))
    env.moveByVel([0, -5, -3])
    print (env.arrive_aim([0,0,-5],5))
    print (distance([0,0,-5],env.getPos()),"threshold:",5)
    print (env.arrive_aim([0,0,-5],25))
    print (distance([0,0,-5],env.getPos()),"threshold:",25)

