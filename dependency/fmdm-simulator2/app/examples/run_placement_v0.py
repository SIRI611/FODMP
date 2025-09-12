import os
import gym
import time
import random
import pathlib
import gym_env
import numpy as np
from PIL import Image

MAX_STEPS = 200
SLEEP_DURATION = 4
RENDER_MODE = 'rgb_array'
PICTURE_DESTINATION = str(pathlib.Path(__file__).absolute().parent.parent / 'sample_pics' / 'placement') + '/'
print(PICTURE_DESTINATION)
if not os.path.isdir(PICTURE_DESTINATION):
    os.makedirs(PICTURE_DESTINATION)

def get_rgb_images_from_array(env, image_num, rgb_pxs):
    """
    Converts the rgb data from array form to image form and saves the images to the location
    specified by PICTURE_DESTINATION. The step number that each image was taken during is specified in the name of each
    image.

    Parameters
    ----------
    env : gym_env.envs.placement_v0.Placement0Env
        Placement task environment.
    image_num : int
        Current step number.
    rgb_px : numpy.ndarray
        Numpy array representing the camera RGB data.
    """
    rgb_imgs = [ Image.fromarray(rgb_px, 'RGB') for rgb_px in rgb_pxs ]
    view = 0
    try:
        for rgb_img in rgb_imgs:
            rgb_img.save('{}step{}_view{}_rgb.png'.format(PICTURE_DESTINATION,image_num,view))
            view+=1
        # Time delay is added to control the number of pictures taken
        time.sleep(SLEEP_DURATION)
    except (IOError or ValueError) as er1:
        env.close()
        print(type(er1))
        print(er1.args[0])
        exit()


def random_pos_within_bin(arm):
    """
    Gets the target joint poses that correspond to the end effector of the Franka Arm being in some random position
    within the scope of the bin.

    Parameters
    ----------
    arm : gym_env.components.franka_arm.FrankaArm

    Returns
    -------
    numpy.ndarray
        Array consisting of the target joint poses for all move-able joints in the arm as calculated by ik, length 9.
    """
    upright_orientation = [0.718887431915408, -0.6933274427520395, -0.03597435521934313, -0.03469528970248905]
    center_of_bin = [0, 0, 0.635]
    xy_offset, z_offset = 0.05, 0.8
    ee_position = [
        random.uniform(center_of_bin[0] - xy_offset, center_of_bin[0] + xy_offset),
        random.uniform(center_of_bin[1] - xy_offset, center_of_bin[1] + xy_offset),
        random.uniform(center_of_bin[2], center_of_bin[2] + z_offset)
    ]
    joint_poses = arm.use_inverse_kinematics(ee_position, upright_orientation, True)
    # Opens fingers
    # joint_poses[7:] = arm.ul[7:]
    joint_poses[7:] = [0,0]
    return joint_poses


def run():
    """
    Steps the custom franka gym environment by applying actions to the franka arm in the environment until a terminating
    condition is met. Then returns the time that it took for the franka arm to reach a terminating state.

    Returns
    -------
    float
        The time that it took for the franka arm to reach a terminating state.
    """
    done = False
    # Gets the start time
    s = time.monotonic()
    f = time.monotonic()
    # Creates franka gym environment
    env = gym.make('placement-v0', config_file=os.environ['path-to-configs'] + 'placement0.yaml')
    observation = env.reset()
    image_num = 0
    while not done:
        get_rgb_images_from_array(env, image_num, observation)
        action = random_pos_within_bin(env.franka_robot)
        observation, reward, done, info = env.step(action)
        image_num += 1
    env.close()
    f = time.monotonic()
    return f - s


if __name__ == '__main__':
    """
    Runs the example.
    """
    print('Time =', run())
