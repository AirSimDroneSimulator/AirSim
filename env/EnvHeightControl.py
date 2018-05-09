from env.EnvBase import *

IMAGE_VIEW = True

class EnvHeightControl(EnvBase):
    def __init__(self, start=[-23, 0, -10], aim=[-23, 125, -10], scaling_factor=2, show_img = True):
        EnvBase.__init__(self)
        self.scaling_factor = scaling_factor
        self.start = np.array(start)
        self.aim = np.array(aim)
        self.height_limit = -30
        self.rand = False
        self.show_img = show_img
        if aim is not None:
            self.aim_height = self.aim[2]
        else:
            self.rand = True
            self.start = np.array([0, 0, -10])

    def reset(self):
        if self.rand:
            self.reset_aim()
        EnvBase.reset(self)
        self.state, _ = self.getState()
        return self.state

    def getState(self):
        pos = v2t(self.client.getPosition())
        vel = v2t(self.client.getVelocity())
        img, mind = self.getImg()
        state = [img, np.array([pos[2] - self.aim_height])]

        return state, mind

    def step(self, action):
        pos = v2t(self.client.getPosition())
        dpos = self.aim - pos

        if abs(action) > 1:
            print("action value error")
            action = action / abs(action)

        temp = np.sqrt(dpos[0] ** 2 + dpos[1] ** 2)
        dx = dpos[0] / temp * self.scaling_factor
        dy = dpos[1] / temp * self.scaling_factor
        dz = - action * self.scaling_factor
        # print (dx,dy,dz)
        EnvBase.moveByDist(self, [dx, dy, dz])

        state_, mind = self.getState()
        pos = state_[1][0]

        info = None
        done = False
        reward = self.rewardf(self.state, state_)

        if self.isDone():
            if self.rand:
                done = False
                reward = 50
                info = "success"
                self.reset_aim()
            else:
                done = True
                reward = 50
                info = "success"

        #if self.client.getCollisionInfo().has_collided:
        if self.collisionCheck(mind) or self.client.getCollisionInfo().has_collided:
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
        norm_state[1] = norm_state[1] / 50

        return norm_state, reward, done, info

    def collisionCheck(self,mind):
        if mind < 0.5:
            return True
        return False

    def isDone(self):
        taim = self.aim
        taim[2] = self.getPos()[2]
        return self.arrive_aim(taim,self.scaling_factor)

    def rewardf(self, state, state_):
        pos = state[1][0]
        pos_ = state_[1][0]
        reward = - abs(pos_) + 5
        reward = reward * 2

        return reward

    def getImg(self):

        responses = self.client.simGetImages(
            [AirSimClient.ImageRequest(0, AirSimClient.AirSimImageType.DepthPerspective, True, False)])
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        mind = img2d.min()
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((64, 64)).convert('L'), dtype=np.float) / 255
        im_final.resize((64, 64, 1))
        if IMAGE_VIEW:
            cv2.imshow("view", im_final)
            key = cv2.waitKey(1) & 0xFF;
        return im_final, mind

if __name__ == "__main__":
    env = EnvHeightControl()
    s = env.reset()
    print (np.max(s[0])*255,np.min(s[0])*255)
    print (env.isDone())
    #rint (env.getImg())
    for i in range(20):
        s,r,_,_ = env.step(0)
        print (np.max(s[0])*255,np.min(s[0])*255)

