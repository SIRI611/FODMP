import numpy as np
from transforms3d.euler import euler2mat
from gym_env.components.picking_robot import PickingRobot
from gym_env.components.simple_franka_robot import SimpleFrankaRobot
from gym_env.utils.helpers import random_val_continuous
from gym_env.utils.placement_utils import qmult


def apply_delta_action(
    bullet_client,
    robot,
    delta_action,
    upper_wksp_bound=0.5,
    lower_wksp_bound=-0.5
):
    """
    Takes the target offset specified by the delta_action parameter and adds it to the current end effector
    position and orientation. Inverse kinematics is then used to calculate the poses of all the robots joints.
    The action is then applied to the robot and the gripper is opened or closed depending on the
    command specified by delta_action. Optionally returns the reaction forces sensed by each end effector
    (if get-end-effector-force-sensor-readings is enabled in the config file).

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    robot : gym_env.components.picking_robot.PickingRobot
        Instance of the PickingRobot class to apply an action to.
    delta_action : list
        [dx, dy, dz, dalpha, open/close gripper]. Specifically the change in position and orientation to apply
        to the gripper of the picking robot as well as whether to open or close the gripper.
    lower_wksp_bound : numpy.ndarray
        Array with shape (3,) consisting of the lower bounds of the workspace in the x, y, and z dimension.
    upper_wksp_bound : numpy.ndarray
        Array with shape (3,) consisting of the upper bounds of the workspace in the x, y, and z dimension.
    Returns
    -------
    any
        If enable-gripper-force-readings is enabled, this method returns a dictionary consisting of a key, value
        pair where each key corresponds to one of the gripper's fingers and the value consists of a list of the
        reaction forces sensed by the respective finger at each simulation step. Otherwise, nothing is returned.
    """
    dx, dy, dz, da, close_gripper = delta_action
    dyaw, dpitch, droll = 0, 0, da
    current_grasptarget_pos, current_gripper_orn, link_state = robot.get_grasptarget_state()
    change_in_orn_euler = np.array([dyaw, dpitch, droll])
    target_gripper_orn = bullet_client.getQuaternionFromEuler(current_gripper_orn + change_in_orn_euler)
    target_grasptarget_pos = current_grasptarget_pos + np.array([dx, dy, dz])
    if robot.clip_delta_actions_to_workspace:
        assert (upper_wksp_bound != None) and (lower_wksp_bound != None), 'An upper and lower workspace bound must ' \
            'be specified if actions are to be clipped to the workspace.'
        target_grasptarget_pos = np.clip(target_grasptarget_pos, a_min=lower_wksp_bound, a_max=upper_wksp_bound)
        print('clipped target position:', target_grasptarget_pos)
    target_joint_poses = robot.use_inverse_kinematics(
        grasptarget_pos=target_grasptarget_pos,
        grasptarget_orn=target_gripper_orn,
        use_null_space=True
    )
    gripper_reaction_forces = robot.apply_action_pos(target_joint_poses)
    if close_gripper > 0 and robot.gripper_is_open:
        robot.close_gripper()
    elif close_gripper < 0 and not robot.gripper_is_open:
        robot.open_gripper()
    if robot.enable_gripper_force_readings:
        return gripper_reaction_forces


def check_collision_between_robot_and_object(bullet_client, robot_id, object_id, collision_threshold=0.0015):
    """
    Given the unique id of a robot and the unique id of the object of interest (i.e. a bin), checks if the two are
    colliding (within a certain threshold that qualifies as a collision.

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    robot_id : int
        Unique id of robot.
    object_id : int
        Unique id of object.
    collision_threshold : float
        The maximum distance that the objects may be apart from each other for them to be considered colliding.
    Returns
    -------
    bool
        True if the robot is colliding with the object, false otherwise.
    """
    distance_between_objects_index = 8
    contact_points = bullet_client.getContactPoints(bodyA=robot_id, bodyB=object_id)
    for contact_point in contact_points:
        distance_between_objects = contact_point[distance_between_objects_index]
        if distance_between_objects < collision_threshold:
            return True
    return False


def check_grasp(bullet_client, robot, objects, min_grasp_height, ensure_stable_grasp):
    """
    For all objects that are above the height specified by min_grasp_height, checks if they are in contact with both
    fingers on the robot's gripper. If so, the grasp is deemed successful unless ensure_stable_grasp is enabled, in
    which case the robot's end effector will shake and the success of the grasp will then be reassessed.

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    robot : gym_env.components.picking_robot.PickingRobot
        Instance of the PickingRobot class.
    objects : list
        A list of integers corresponding to the unique ids of objects to consider for grasp-checking. 
    min_grasp_height : float
        The minimum height objects must be at for this function to check if they are being grasped by the robot
        (ideally should be some distance above the structure that is holding the objects).
    ensure_stable_grasp : bool
        If true, upon a successful grasp the robot's end effector will shake and the success of the grasp will then be
        reassessed. Otherwise this function will deem the grasp successful if the object is above the min_grasp_height
        and in contact with both fingers on the robot's gripper.
    Returns
    -------
    tuple
        A bool and an int. The bool will be true or false depending on whether the robot is successfully grasping an
        object and the int will be the unique id of the object being grasped.
    """
    position_index, z_index = 0, 2
    for obj_id in objects:
        obj_position = bullet_client.getBasePositionAndOrientation(obj_id)[position_index]
        if obj_position[z_index] >= min_grasp_height:
            if robot.is_grasping_object(obj_id) and ensure_stable_grasp:
                robot.shake()
                if robot.is_grasping_object(obj_id):
                    return True, obj_id
            elif robot.is_grasping_object(obj_id):
                return True, obj_id
    return False, None


def load_robot(bullet_client, robot_config, finger_indices, grasptarget_index):
    """
    Gets an instance of the PickingRobot class according to the configuration specified by the parameter robot_config.
    
    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    robot_config : dict
        The robot section of the config file pertaining to the instance of the PickingRobot class that is to be
        created.
    finger_indices : list
            The index of each finger that makes up the gripper of the picking robot, length 2.
    grasptarget_index : int
        The index of the grasptarget (the location between the fingers of the gripper where an object would ideally
        be if it were to be grasped) - this index is considered the "end effector" in inverse kinematics
        calculations.
    Returns
    -------
    gym_env.components.picking_robot.PickingRobot
        Instance of the PickingRobot class.
    """
    return SimpleFrankaRobot(
        bullet_client,
        robot_config,
        finger_indices,
        grasptarget_index
    )


def mount_camera_on_robot(robot, camera):
    """
    Adjusts the view matrix of the camera so it is as if the camera is mounted on the end effector of the robot.

    Parameters
    ----------
    robot : gym_env.components.picking_robot.PickingRobot
        Instance of the PickingRobot class.
    camera : gym_env.components.camera.Camera
        Instance of the Camera class.
    """
    i, j, k = 0, 1, 2
    position, orientation_euler, _ = robot.get_grasptarget_state()
    orientation_matrix = euler2mat(*orientation_euler)
    target_position = position.copy() + (0.08 * orientation_matrix[:, k])
    position -= 0.15 * orientation_matrix[:, k]
    position += 0.1 * orientation_matrix[:, i]
    camera.adjust_view_matrix(position, target_position, orientation_matrix[:, j])

def reset_arm(robot, robot_config):
    """
    Attempts to move the arm grasp target to the position and orientation set in the config file, if it is configured
    to set the arm state from a target position.
    Parameters
    ----------
    robot : gym_env.components.picking_robot.PickingRobot
        Instance of the PickingRobot class.
    robot_config : dict
        Dictionary corresponding to the robot section of the config file for the poses to be set from.
    """
    try:
        config = robot_config['poses']['from-grasptarget-state']
    except:
        config = {'enabled':False}
    if config['enabled']:
        position = [ random_val_continuous(pos) for pos in config['position'] ]
        position[2] = 0.05  # set z height to 0.15
        default_orientation = config['default-orientation']
        try:
            orientation = [  np.radians(random_val_continuous(orn)) for orn in config['orientation'] ]
            orientation = robot.bc.getQuaternionFromEuler(orientation)
            orientation = qmult(orientation,default_orientation)
        except:
            raise("Something is wrong with the orientation config for robot:poses:from-grasptarget-state")
        joint_poses = robot.use_inverse_kinematics(position, orientation, False)
        robot.default_poses = joint_poses
        robot.reset()
