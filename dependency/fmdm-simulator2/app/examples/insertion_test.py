import os
import gym
import time
import random
import pathlib
import gym_env
import numpy as np
from PIL import Image

EPISODES = 100000
NUM_STEPS = 80

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
    """
    Runs the example.
    https://frankaemika.github.io/docs/control_parameters.html
    The three control modes in Franka Panda Arm:
    joint velocity: 
        https://github.com/frankaemika/franka_ros/blob/kinetic-devel/franka_example_controllers/src/joint_velocity_example_controller.cpp
    torques: 
        https://github.com/frankaemika/franka_ros/blob/kinetic-devel/franka_example_controllers/src/force_example_controller.cpp
    joint_impedance: 
        https://github.com/frankaemika/franka_ros/blob/kinetic-devel/franka_example_controllers/src/joint_impedance_example_controller.cpp
    tutorial from Peter Corke: https://petercorke.com/robotics/franka-emika-control-interface-libfranka/
    Example code from Visp: joint velocity, Cart Velocity, Joint Torques
        https://visp-doc.inria.fr/doxygen/visp-daily/vpRobotFranka_8cpp_source.html
    Other example codes: only joint pose control and dx, dy, dz control:
        https://github.com/nebbles/DE3-ROB1-CHESS/blob/master/franka/franka_control_ros.py
        https://github.com/frankaemika/icra18-fci-tutorial/blob/master/icra18/scripts/demo.py
    Franka robot maximum velocities: [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, 0.2, 0.2]
    saturated velocity: filter out big velocities.
    """
    """
        Steps the custom franka gym environment by applying actions to the franka arm in the environment until a terminating
        condition is met. Then returns the time that it took for the franka arm to reach a terminating state.

        Returns
        -------
        float
            The time that it took for the franka arm to reach a terminating state.
        """
    # Gets the start time
    s = time.monotonic()
    f = time.monotonic()
    # Creates franka gym environment
    env = gym.make('placement-v1', config_file=os.environ['path-to-configs'] + 'placement2.yaml')
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
