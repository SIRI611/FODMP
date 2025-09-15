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
PICTURE_DESTINATION = str(pathlib.Path(__file__).parent.parent / 'sample_pics') + '/'
# Set this to the same value as camera > scale-depth-data in config file
SCALED_DEPTH_DATA = False


def auto_scale_depth_array(depth_array):
    """
    Scales the depth data so that it uses as much of the range between 0-255 as possible. This makes the camera depth
    image much easier to see than when the depth data is scaled by a static factor of 255. Thus, this functionality is
    handy but not ideal to have built in to the environment because when auto-scaling is used the scale factor would
    likely change with each iteration making it extremely difficult for an agent to learn anything from the data. That
    is why this is done outside of the gym environment as an example implementation.

    Parameters
    ----------
    depth_array : numpy.ndarray
        Numpy array representing the camera depth data with elements that have values between 0 and 1.

    Returns
    -------
    numpy.ndarray
        Scaled numpy array representing the camera depth data with elements that have values between 0 and 255. Ideally,
        the elements should use most of this range.
    """
    min_scale_factor = 255
    max_elem_row, max_elem_col = np.unravel_index(depth_array.argmax(), depth_array.shape)
    scale_factor = min_scale_factor / depth_array[max_elem_row][max_elem_col]
    return (depth_array * scale_factor).astype(np.uint8)


def get_rgb_depth_images_from_array(env, image_num, rgb_px, depth_px):
    """
    Converts the rgb data and the depth data from array form to image form and saves the images to the location
    specified by PICTURE_DESTINATION. The step number that each image was taken during is specified in the name of each
    image.
    IMPORTANT: make sure that camera > scale-depth-data in the config file has the same value as the global variable
    SCALED_DEPTH_DATA as if it doesnt, the depth images will not turn out right.

    Parameters
    ----------
    env : gym_env.envs.frank_v0.Franka0Env
        Franka Gym environment.
    image_num : int
        Current step number.
    rgb_px : numpy.ndarray
        Numpy array representing the camera RGB data.
    depth_px : numpy.ndarray
        Numpy array representing the camera depth data.
    """
    assert env.config_file['camera']['scale-depth-data'] == SCALED_DEPTH_DATA, \
        'Please ensure that camera > scale-depth-data in the configuration file has the same value as the global ' \
        'variable SCALED_DEPTH_DATA in this file (run_franka0env.py), as if it does not the depth images will not ' \
        'turn out properly.'
    if not SCALED_DEPTH_DATA:
        depth_px = auto_scale_depth_array(depth_px)
    rgb_img = Image.fromarray(rgb_px, 'RGB')
    depth_img = Image.fromarray(depth_px)
    try:
        rgb_img.save(PICTURE_DESTINATION + '/sample_pics/step' + str(image_num) + 'rgb.png')
        depth_img.save(PICTURE_DESTINATION + '/sample_pics/step' + str(image_num) + 'depth.png')
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
    center_of_bin = [0, 0, 0.11]
    xy_offset, z_offset = 0.05, 0.4
    ee_position = [
        random.uniform(center_of_bin[0] - xy_offset, center_of_bin[0] + xy_offset),
        random.uniform(center_of_bin[1] - xy_offset, center_of_bin[1] + xy_offset),
        random.uniform(center_of_bin[2], center_of_bin[2] + z_offset)
    ]
    joint_poses = arm.use_inverse_kinematics(ee_position, upright_orientation, True)
    # Opens fingers
    # joint_poses[7:] = arm.ul[7:]
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
    env = gym.make('franka-v0', config_file=os.environ['path-to-configs'] + 'franka0.yaml')
    observation = env.reset()
    image_num = 0
    while not done:
        get_rgb_depth_images_from_array(env, image_num, observation[0], observation[1])
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
