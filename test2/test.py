# import gym
# import gymEnv
# print(gymEnv.__file__)

# env = gym.make('gymEnv:franka')

import gym
env = gym.make('FrankaReacher-v0')
env.reset()
env.render()
while True:
    print(1)