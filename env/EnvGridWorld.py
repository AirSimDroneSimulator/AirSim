from env.EnvBase import *

class EnvGridWorld(EnvBase):
    def __init__(self, start=[0, 0, -5], aim=[32, 38, -4], scaling_factor=5):
        EnvBase.__init__(self)
        self.start = np.array(start)
        self.aim = np.array(aim)
        self.scaling_factor = scaling_factor
        self.timestep = 1
        if aim == None:
            self.test = True
        else:
            self.test = False

    def interpret_action(self, action):
        scaling_factor = self.scaling_factor
        quad_offset = (0, 0, 0)
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

    def step(self, action):
        diff = self.interpret_action(action)
        EnvBase.moveByDist(self, diff)

        state_ = self.getState()
        pos_ = self.getPos()
        vel_ = self.getVel()

        info = None
        done = False
        reward = self.rewardf(self.state, state_)
        reawrd = reward / 50
        if action == 0:
            reward -= 10
        if self.arrive_aim(self.aim,self.scaling_factor):
            done = True
            reward = 100
            info = "success"
            if self.test:
                done = False
                self.reset_aim()
        if self.client.getCollisionInfo().has_collided:
            reward = -100
            done = True
            info = "collision"
        if (distance(pos_, self.aim) > 150):
            reward = -100
            done = True
            info = "out of range"

        self.state = state_

        return state_, reward, done, info

    def reset(self):
        EnvBase.reset(self)
        if self.test:
            self.reset_aim()
        state = self.getState()
        self.state = state
        return state

    def getState(self):
        pos = self.getPos()
        vel = self.getVel()
        state = np.append(pos, vel)
        aim = self.aim
        state  = np.append(state, aim)
        return state


    def rewardf(self, state, state_):

        dis = distance(state[0:3], self.aim)
        dis_ = distance(state_[0:3], self.aim)
        reward = dis - dis_
        reward = reward * 1
        reward -= 1
        return reward

if __name__ == "__main__":
    env = EnvGridWorld()
    env.reset()
    print (env.interpret_action(2),env.interpret_action(6),env.interpret_action(7))
    print (env.step(1))
    print (env.step(6))
