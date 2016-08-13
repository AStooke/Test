import gym
import time
env = gym.make('Hopper-v1')
env.reset()
time.sleep(.1)
for _ in xrange(100):
    env.step(env.action_space.sample()) # take a random action
    env.render()
    time.sleep(.1)
