import os
import gym
import time
import random
import pathlib
import gym_env
import numpy as np
from PIL import Image

EPISODES = 100
NUM_STEPS = 100
SEED = 112  # random.randint(0,99999999999)
# 0.707, -0.707, 0, 0
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    # Gets the start time
    s = time.monotonic()
    f = time.monotonic()
    # Creates franka gym environment
    env = gym.make('ball_into_goal-v0', config_file=os.environ['path-to-configs'] + 'ball_into_goal.yaml')
    sample_collection_buffer = []
    env_step_count = 0
    # obs = env.reset()
    # print(obs['state'])
    success_count = 0
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
            # print(done)
            if info['success']:
                success_count += 1

            if done:
                print('finished in {} steps'.format(i))
                break

    print(' ball_into_goal success rate {}%'.format(success_count))
    env.close()
    f = time.monotonic()

