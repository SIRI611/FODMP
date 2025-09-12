import os
import gym
import time
import random
import gym_env
import numpy as np
from PIL import Image


def act(obs, info, **kwargs):
    if np.random.uniform() < 0.:
        action = np.random.uniform(-np.ones(4), np.ones(4))
        gripper = np.random.choice([-1, 1], p=[0.5, 0.5], size=(1,))
        end = np.random.choice([-1, 1], p=[0.8, 0.2], size=(1,))
        action = np.concatenate((action, gripper, end))
        return action

    pos, grasp_success = obs['grasptarget'], bool(info['grasp_success'])
    gripper = pos[-1]

    if pos[2] < 0.03:  # time to grasp object
        if gripper > 0:
            np.array([0, 0, 1, 0, 1, -1], dtype=np.float32)
            # print("Go Up")
        else:
            # print("Closing Gripper")
            return np.array([0, 0, 0, 0, 1, -1], dtype=np.float32)
        
    if grasp_success:  # if any objected is lifted up successfully, end episode
        # print("Ending Episode")
        return np.array([0, 0, 0, 0, 1, 1], dtype=np.float32)

    if gripper > 0  and pos[2] < 0.13:  # if gripper closed and below certain height, go up
        # print("Go up")
        return np.array([0, 0, 1, 0, 1, -1], dtype=np.float32)
    
    # print("Go Down")
    action = np.random.uniform(-np.ones(4), np.ones(4))
    action[2] = -1
    action = np.concatenate((action, np.asarray([-1, -1])))
    return action


def run(env):
    """
    Steps the custom franka gym environment by applying actions to the franka arm in the environment until a terminating
    condition is met. Then returns the time that it took for the franka arm to reach a terminating state.

    Parameters
    ----------
    env : gym_env.envs.franka_v2.FrankaEnv2
        An instance of the franka-v2 custom gym environment. 
    Returns
    -------
    float
        The time that it took for the franka arm to reach a terminating state.
    """
    done = False
    # Gets the start time
    s = time.monotonic()
    f = time.monotonic()
    observation, info = env.reset()
    while not done:
        action = act(observation, info)
        observation, reward, done, info = env.step(action)
        if done:
            print('Reward ', reward)
    f = time.monotonic()
    return f - s


if __name__ == '__main__':
    """
    Runs the example.
    """
    # Creates franka gym environment
    env = gym.make('franka-v2', config_file=os.environ['path-to-configs'] + 'franka2.yaml')
    runs = 100
    total_time = 0
    for _ in range(runs):
        total_time += run(env)
    env.close()
    print('Average runtime =', total_time/runs)
