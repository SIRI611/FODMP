import os
import gym
import time
import random
import pathlib
import gym_env
import numpy as np
from PIL import Image

EPISODES = 100000
NUM_STEPS = 800

def random_joint_vel(max_joint_vel):
    """
    random joint velocities.
    :param arm:
    :return:
    """
    scale = 0.1
    return [random.uniform(-max_vel*scale, max_vel*scale) for max_vel in max_joint_vel]

def vis_obs(obs):
    obs_cam1, obs_cam2 = obs
    Image.fromarray(obs_cam1.astype('uint8'), 'RGB').save('cam1.jpg')
    Image.fromarray(obs_cam2.astype('uint8'), 'RGB').save('cam2.jpg')

# 0.707, -0.707, 0, 0

if __name__ == '__main__':
    # Gets the start time
    s = time.monotonic()
    f = time.monotonic()
    # Creates franka gym environment
    env = gym.make('placement-v4', config_file=os.environ['path-to-configs'] + 'placement4.yaml')
    max_joint_velocities = [2.175, 2.175, 2.175, 2.175, 2.61, 2.61]  # 2.61
    # policy --> joint velocity --> picking_robot.apply_action_vel
    #                                    # check max, saturated
    #                                    # add internal loop running at 100Hz, v_apply = v_input - sigma * torques
    for m in range(EPISODES):
        observation = env.reset()
        # vis_obs(observation)
        for i in range(NUM_STEPS):
            step_s = time.monotonic()  # start time of the step
            action = random_joint_vel(max_joint_velocities)
            print('==episode {:4d}/{}, step {:4d}/{}, action: ['.format(m, EPISODES, i, NUM_STEPS), end='')
            for k in action:
                print('{:+.2f} '.format(k), end='')
            observation, reward, done, info = env.step(action)
            print('] Duration: {:2.3f}s, Reward: {:+.2f}  Total time spent: {:4.1f}]'.format(
                time.monotonic() - step_s,reward, time.monotonic() - s))

    env.close()
    f = time.monotonic()
