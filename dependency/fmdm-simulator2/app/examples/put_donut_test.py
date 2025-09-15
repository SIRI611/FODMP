import os
import gym
import time
import random
import pathlib
import gym_env
import numpy as np
from PIL import Image

EPISODES = 100000
NUM_STEPS = 100

# 0.707, -0.707, 0, 0

if __name__ == '__main__':
    # Gets the start time
    s = time.monotonic()
    f = time.monotonic()
    # Creates franka gym environment
    env = gym.make('put_donut-v0', config_file=os.environ['path-to-configs'] + 'put_donut.yaml')
    sample_collection_buffer = []
    env_step_count = 0
    # obs = env.reset()
    # print(obs['state'])
    for m in range(EPISODES):
        obs = env.reset()
        print('reset')
        # vis_obs(observation)
        for i in range(NUM_STEPS):
            step_s = time.monotonic()  # start time of the step
            action = env.expert_action_one_robot()
            # print('==episode {:4d}/{}, step {:4d}/{}, action: ['.format(m, EPISODES, i, NUM_STEPS), end='')
            # for k in action:
            #     print('{:+.2f} '.format(k), end='')
            observation, reward, done, info = env.step(action)
            # print('] Duration: {:2.3f}s, Reward: {:+.2f}  Total time spent: {:4.1f}]'.format(
            #     time.monotonic() - step_s,reward, time.monotonic() - s))
            print(done)
            if done:
                break
    env.close()
    f = time.monotonic()

