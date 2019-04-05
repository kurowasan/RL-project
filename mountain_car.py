import math

import numpy as np
import gym
import gym.envs.classic_control.mountain_car

def get_modified_MountainCarEnv():

    def _height(self, xs):
        return 0.05 / (xs + 1.2 + 1e-16) + np.sin(3 * xs) * .45 + .5

    def step(self, action):
        # only two lines are modified
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        # 1st modification
        velocity += (action - 1) * self.force + (-0.1 * (position + 1.2)**(-2) + math.cos(3 * position)) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        # 2nd modification
        #if (position == self.min_position and velocity < 0): velocity = 0

        done = bool(position >= self.goal_position)
        reward = -1.0

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    gym.envs.classic_control.mountain_car.MountainCarEnv._height = _height
    gym.envs.classic_control.mountain_car.MountainCarEnv.step = step

    return gym.make("MountainCar-v0")


if __name__ == "__main__":
    env = get_modified_MountainCarEnv()
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action
    env.close()

