from env.EnvBase import *

IMAGE_VIEW = True

class EnvImgControl(EnvBase):
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
            self.aim_height = self.aim[2]
        EnvBase.reset(self)
        self.state, _ = self.getState()
        return self.state

    def getState(self):
        img, mind = self.getImg()
        state = img
        self.mind = mind
        return state, mind

    def step(self, action):
        pos = v2t(self.client.getPosition())
        dpos = self.aim - pos

        self.action = action

        if action == 1:
            action = 1
        elif action == 0:
            action = -dpos[2]
        #elif action == 2:
        #    action = -1

        if abs(action) > 1:
            #print("action value error")
            action = action / abs(action)

        temp = np.sqrt(dpos[0] ** 2 + dpos[1] ** 2)
        dx = dpos[0] / temp * self.scaling_factor
        dy = dpos[1] / temp * self.scaling_factor
        dz = - action * self.scaling_factor
        # print (dx,dy,dz)
        EnvBase.moveByDist(self, [dx, dy, dz])

        state_, mind = self.getState()

        info = None
        done = False
        reward = self.rewardf(self.state, state_)

        if self.isDone():
            if self.rand:
                done = False
                reward = 100
                info = "success"
                self.reset_aim()
                self.aim_height = self.aim[2]
            else:
                done = True
                reward = 100
                info = "success"

        #if self.client.getCollisionInfo().has_collided:
        if self.collisionCheck(mind) or self.client.getCollisionInfo().has_collided:
            reward = -100
            done = True
            info = "collision"
        if (pos[2]) < self.height_limit:
            done = True
            info = "too high"
            reward = -100

        self.state = state_
        norm_state = copy.deepcopy(state_)

        return norm_state, reward, done, info

    def collisionCheck(self,mind):
        if mind < 0.2:
            return True
        return False

    def isDone(self):
        taim = self.aim.copy()
        taim[2] = self.getPos()[2]
        return self.arrive_aim(taim,self.scaling_factor)

    def rewardf(self, state, state_):
        threshold = 2
        reward1 = -self.action * 2 + 1
        reward1 = reward1 + 1
        reward1 = reward1 / 2

        reward2 = min(self.mind,threshold)
        reward2 = reward2 * 2
        reward2 = reward2 / 2

        reward = reward1 + reward2
        return reward

    def getImg(self):

        responses = self.client.simGetImages(
            [AirSimClient.ImageRequest(0, AirSimClient.AirSimImageType.DepthPerspective, True, False),
             AirSimClient.ImageRequest(3, AirSimClient.AirSimImageType.DepthPerspective, True, False),
             AirSimClient.ImageRequest(4, AirSimClient.AirSimImageType.DepthPerspective, True, False)])
        im_res = []
        mind = 100
        for i in range(len(responses)):
            response = responses[i]
            img1d = np.array(response.image_data_float, dtype=np.float)
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
            mind = min(img2d.min(),mind)
            image = Image.fromarray(img2d)
            im_final = np.array(image.resize((64, 64)).convert('L'), dtype=np.float) / 255
            im_final.resize((64, 64, 1))
            if IMAGE_VIEW:
                cv2.imshow("view"+str(i), im_final)
                key = cv2.waitKey(1) & 0xFF;
            im_res.append(im_final)
        temp = [m for m in im_res]
        res = np.concatenate(temp,axis = -1)
        return res, mind

if __name__ == "__main__":
    env = EnvImgControl()
    s = env.reset()
    s, r, d, i = env.step(0)
    _, mind = env.getImg()
    print(mind)
    for i in range(5):
        s,r,d,i = env.step(1)
        _,mind = env.getImg()
        print (mind)

    for i in range(10):
        s, r, d, i = env.step(0)
        print(env.aim - env.getPos(),env.getPos(),env.aim)


