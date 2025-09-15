import os
from gym_env.envs.franka_v0 import FrankaEnv0

DEFAULT_POSITION = [0, 0, 0.4]
DEFAULT_ORIENTATION = [0.718887431915408, -0.6933274427520395, -0.03597435521934313, -0.03469528970248905]
ENABLE_ADDITIONAL_GUI_FEATURES = 0
EE_INDEX = 11
ARM_DOF = 7
O = 0.025
C = 0


def check_key_events(env, position, orientation, ee_o):
    """
    Gets all the keyboard events that have happened since getKeyboardEvents() was last called and then adjusts any
    state settings that were adjusted (by a key being pressed and returns the updated info). If nothing has been changed
    the current position of the arm is maintained.

    Parameters
    ----------
    env : gym_env.envs.franka_v0.Franka0Env
        Current working environment.
    position : list
        Current position of end effector, length 3.
    orientation : list
        Current orientation of end effector, length 4.
    ee_o : bool
        True if end effectors are currently open, False if they are currently closed.
    Returns
    -------
    any
        Either a string or a list depending on the key pressed. 'reset' is returned if the reset key (r) has been
        pressed, 'toggle ee' is returned if the toggle end effector key (space) has been pressed, a list containing the
        updated position of the arm's end effector (length 3) if a position control key has been pressed, or a list
        containing the updated orientation of the arm's end effector (length 4) if an orientation control key has been
        pressed.
    """
    inc, space_key = 0.001, 32
    outcome = []
    pos_key_consequences = {
        ord('1'): (0, inc),
        ord('2'): (0, -inc),
        ord('3'): (1, inc),
        ord('4'): (1, -inc),
        ord('5'): (2, inc),
        ord('6'): (2, -inc)
    }
    orn_key_consequences = {
        ord('7'): (0, inc),
        ord('8'): (0, -inc),
        ord('9'): (1, inc),
        ord('0'): (1, -inc),
        env.bullet_client.B3G_RIGHT_ARROW: (2, inc),
        env.bullet_client.B3G_LEFT_ARROW: (2, -inc),
        env.bullet_client.B3G_UP_ARROW: (3, inc),
        env.bullet_client.B3G_DOWN_ARROW: (3, -inc)
    }
    other_consequences = {
        space_key: 'toggle ee',
        ord('r'): 'reset'
    }
    key_dict = env.bullet_client.getKeyboardEvents()
    for key in pos_key_consequences:
        if (key in key_dict) and (key_dict[key] & env.bullet_client.KEY_IS_DOWN):
            position = update_coords(position, pos_key_consequences[key][0], pos_key_consequences[key][1])
    for key in orn_key_consequences:
        if (key in key_dict) and (key_dict[key] & env.bullet_client.KEY_IS_DOWN):
            orientation = update_orn(orientation, orn_key_consequences[key][0], orn_key_consequences[key][1])
    interactive_ik(env, position, orientation, ee_o)
    for key in other_consequences:
        if (key in key_dict) and (key_dict[key] & env.bullet_client.KEY_WAS_TRIGGERED):
            outcome.append(other_consequences[key])
    return outcome


def enable_gui(bc):
    """
    Enables additional features of GUI display (visible axis, wireframe view, etc.).

    Parameters
    ----------
    bc : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    """
    bc.configureDebugVisualizer(bc.COV_ENABLE_GUI, ENABLE_ADDITIONAL_GUI_FEATURES)


def interactive_ik(env, target_pos, target_orn, ee_o):
    """
    Gets the joint poses such that the end effector of the arm in the environment (assumes only one arm in env since
    Franka0Env supports only one arm) is in the specified position with the specified orientation and then moves the
    arm into the target pose. Depending on the flag ee_o, the end effectors are moved into an opened or closed position.

    Parameters
    ----------
    env : gym_env.envs.franka_v0.Franka0Env
        Current working environment.
    target_pos : list
        Target position of end effector, length 3.
    target_orn : list
        Target orientation of end effector, length 4.
    ee_o : bool
        True if end effectors are currently open, False if they are currently closed.
    """
    joint_poses = env.franka_robot.use_inverse_kinematics(
        grasptarget_pos=target_pos,
        grasptarget_orn=target_orn,
        use_null_space=True
    )
    joint_poses[ARM_DOF:] = ([C, C] if not ee_o else [O, O])
    rxn_forces = env.franka_robot.apply_action_pos(joint_poses)


def print_helper_info():
    """
    Prints helpful info regarding useful controls to standard out.
    """
    print('\n\n\nWelcome to Interactive Franka Mode')
    print('\nUse the following keys to configure the gui:')
    print('\tw key -> toggles wireframe viewing mode')
    print('\tc key -> toggles showing current collisions (only visible in wireframe viewing mode)')
    print('\tk key -> toggles showing axis of freedom of move-able joints (only visible in wireframe viewing mode)')
    print('\tj key -> toggles showing axis of orientation of all objects (only visible in wireframe viewing mode)')
    print('\tg key -> enables additional gui features')
    print('\ts key -> toggles shadows and planar reflection (not visible in wireframe viewing mode)')
    print('\nUse the following keys to adjust the position of the arm\'s end effector:')
    print('\t1/2 key -> adjusts x position of end effector forwards/backwards')
    print('\t3/4 key -> adjusts y position of end effector right/left')
    print('\t5/6 key -> adjusts z position of end effector up/down')
    print('\nUse the following keys to adjust the orientation of the arm\'s end effector:')
    print('\t7/8 key -> increase/decrease zeroth element of quaternion')
    print('\t9/0 key -> increase/decrease first element of quaternion')
    print('\tright/left arrow key -> increase/decrease second element of quaternion')
    print('\tup/down arrow key -> increase/decrease third element of quaternion')
    print('\nUse the following keys for additional functionality:')
    print('\tspace key -> toggles end effectors open/closed')
    print('\tr key -> resets environment')
    print('\tescape key -> exits\n')


def run():
    """
    Runs the interactive session. Gets an instance of a Franka0Env and declares necessary environment variables then
    enters an infinite while loop (or at least until esc key or exit is pressed) and continuously checks which keys
    have been pressed, gets the updated state info, and applies the relevant action.
    """
    env, target_x, target_y, target_z, orn_0, orn_1, orn_2, orn_3 = setup()
    enable_gui(env.bullet_client)
    pos = [target_x, target_y, target_z]
    orn = [orn_0, orn_1, orn_2, orn_3]
    ee_open, done = True, False
    print_helper_info()
    while not done:
        done, info = env.check_termination()
        res = check_key_events(env, pos, orn, ee_open)
        if len(res) > 0:
            if 'toggle ee' in res:
                ee_open = not ee_open
            if 'reset' in res:
                ee_open = True
                pos = [target_x, target_y, target_z]
                orn = [orn_0, orn_1, orn_2, orn_3]
                env.reset()
                enable_gui(env.bullet_client)
    print('A TERMINATION CONDITION HAS BEEN MET\n' + str(info) + '\n\n')


def setup():
    """
    Creates an instance of Franka0Env and returns the instance along with the arm's (Franka0Env is a one arm env only)
    end effector's starting position.

    Returns
    -------
    tuple
        An instance of Franka0Env which is a custom gym environment, as well as three floats which correspond to the
        starting x, y, and z position of the end effector of the arm in the environment.
    """
    x, y, z = 0, 1, 2
    env = FrankaEnv0(config_file=os.environ['path-to-configs'] + 'interactive_franka.yaml')
    assert env.config_file['simulation']['sim-mode'] == 'gpu-gui', 'Interactive mode requires gui mode to be enabled.'
    env.reset()
    return env, DEFAULT_POSITION[x], DEFAULT_POSITION[y], DEFAULT_POSITION[z], DEFAULT_ORIENTATION[0], \
           DEFAULT_ORIENTATION[1], DEFAULT_ORIENTATION[2], DEFAULT_ORIENTATION[3]


def update_coords(current_pos, index_to_change, change_by):
    """
    Modifies either the x, y, or z coordinate of the current position of the arm's end effector by the specified amount.

    Parameters
    ----------
    current_pos : list
        Current position of end effector, length 3.
    index_to_change : int
        Correlates to coordinate to modify - in range [0,2].
    change_by : float
        Quantity to change the specified coordinate by.
    Returns
    -------
    list
        Updated end effector position, length 3.
    """
    assert index_to_change in [0, 1, 2], 'Index to change is out of range for list of size 3.'
    current_pos[index_to_change] += change_by
    return current_pos


def update_orn(current_orn, index_to_change, change_by):
    """
    Modifies either the x, y, or z coordinate of the current position of the arm's end effector by the specified amount.

    Parameters
    ----------
    current_orn : list
        Current orientation of end effector, length 4.
    index_to_change : int
        Correlates to coordinate to modify - must be in in range [0,3].
    change_by: float
        Quantity to change the specified index of the quaternion by.
    Returns
    -------
    list
        Updated end effector orientation, length 4.
    """
    assert index_to_change in [0, 1, 2, 3], 'Index to change is out of range for list of size 4.'
    current_orn[index_to_change] += change_by
    return current_orn


if __name__ == '__main__':
    run()
