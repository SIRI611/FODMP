import pytest
import random
import numpy as np


@pytest.mark.base_functionality
class TestFrankaArm:
    def test_read_joint_positions(self, env_instance, config, num_franka_joints):
        """
        Tests the read_joint_positions method of FrankaArm. Ensures that the default joint position instance attribute is
        as specified in the config file (for each arm in the environment).
    
        Parameters
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        config : dict
            File consisting of simulation config.
        num_franka_joints : int
            Fixed number of joints in Franka arm.
        """
        for i in range(len(env_instance.robots)):
            actual_default_joint_positions = env_instance.robots[i].default_joint_positions
            assert actual_default_joint_positions == read_default_joint_positions_from_config(
                config, 'arm-' + str(i), num_franka_joints)
    
    def test_check_target_position_bounds(self, env_instance):
        """
        Tests the check_target_position_bounds method of FrankaArm. Passes target positions that are larger than the upper
        bound and smaller than the lower bound for each end effector and ensures that they get filtered to the max or min
        value (respectively) as specified in the urdf for the arm. This is done for each arm in the environment.
    
        Parameters
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        """
        ee_position_too_large = [1, 1]
        ee_position_too_small = [-1, -1]
        upper_ee_limits = (0.04, 0.04)
        lower_ee_limits = (0, 0)
        for i in range(env_instance.num_arms):
            arm = env_instance.robots[i]
            filtered_value_high = arm.check_target_position_bounds(ee_position_too_large[0], ee_position_too_large[1])
            filtered_value_low = arm.check_target_position_bounds(ee_position_too_small[0], ee_position_too_small[1])
            assert filtered_value_high == upper_ee_limits
            assert filtered_value_low == lower_ee_limits
    
    def test_null_space_limits(self, env_instance, config, num_franka_joints):
        """
        Tests the null_space_limits method of FrankaArm. Ensures that the lower and upper limits are as specified in the
        urdf and that the null space limits are as specified by the default joint poses from the config file. Does this
        for each arm in the environment.
    
        Parameters
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        config : dict
            File consisting of simulation config.
        num_franka_joints : int
            Fixed number of joints in Franka arm.
        """
        for i in range(env_instance.num_arms):
            arm = env_instance.robots[i]
            upper_limits_from_urdf = [2.9671, 1.8326, 2.9671, 0, 2.9671, 3.8223, 2.9671, 0.04, 0.04]
            lower_limits_from_urdf = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671, 0, 0]
            joint_ranges_from_urdf = [upper_limits_from_urdf[j] - lower_limits_from_urdf[j] for j in
                                      range(len(upper_limits_from_urdf))]
            assert arm.upper_limits == upper_limits_from_urdf
            assert arm.lower_limits == lower_limits_from_urdf
            assert arm.joint_ranges == joint_ranges_from_urdf
            assert arm.rest_poses == read_default_joint_positions_from_config(config, 'arm-' + str(i), num_franka_joints)
    
    def test_apply_action_pos(self, env_instance, config):
        """
        Tests the apply_action_pos method of FrankaArm. Moves the arm to a new position and then ensures that the actual
        position of each joint in the arm is within an acceptable proximity (0.0008 radians) of the target position, then
        resets the environment. Does this for each arm in the environment.
    
        Parameters
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        config : dict
            File consisting of simulation config.
        """
        env_instance.reset()
        acceptable_proximity = 0.0008
        for i in range(env_instance.num_arms):
            _, target = shuffle_arm_position(config, env_instance.robots[i], 'arm-'+str(i))
            actual_pose = env_instance.robots[i].get_position_velocity_state()[:9]
            assert target.size == actual_pose.size
            difference = abs(actual_pose - target)
            assert np.array([proximity < acceptable_proximity for proximity in difference]).all()
            env_instance.reset()
    
    def test_apply_action_vel(self, env_instance):
        """
        Tests the apply_action_vel method of FrankaArm. Applies a velocity to each joint in the arm and then checks to
        ensure that there has been a change in velocity in each joint. Does this for each arm in the environment.
    
        Parameters
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        """
        env_instance.reset()
        for i in range(env_instance.num_arms):
            initial_vel = env_instance.robots[i].get_position_velocity_state()[9:]
            shuffle_arm_velocity(env_instance.robots[i])
            actual_vel = env_instance.robots[i].get_position_velocity_state()[9:]
            difference = abs(actual_vel - initial_vel)
            assert np.array([velocity_change > 0 for velocity_change in difference]).all()
            env_instance.reset()
    
    def test_get_position_velocity_state(self, env_instance, config):
        """
        Tests the get_position_velocity method of FrankaArm. Shuffles the position of the arm and then gets the actual
        position of each move-able joint in the arm. Then compares these correct positions to the ones returned by
        get_position_velocity_state to ensure that method is getting the correct joints positions. Does this for each
        arm in the environment.
    
        Parameters
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        config : dict
            File consisting of simulation config.
        """
        env_instance.reset()
        for i in range(env_instance.num_arms):
            arm = env_instance.robots[i]
            shuffle_arm_position(config, arm, 'arm-'+str(i))
            info_on_all_joints = env_instance.bullet_client.getJointStates(arm.arm_id, arm.moveable_joints)
            correct_observation = np.array([info_on_all_joints[j][0] for j in range(len(arm.moveable_joints))] +
                                           [info_on_all_joints[k][1] for k in range(len(arm.moveable_joints))])
            assert (arm.get_position_velocity_state() == correct_observation).all()
    
    def test_get_joint_poses_ik(self, env_instance, config):
        """
        Tests the get_joint_poses_ik method of FrankaArm. The end effector of the arm is moved to a random position and then
        the final position of the end effector is compared to the target position and if it is within an acceptable
        proximity then the test is passed and the environment is reset. This is repeated for each arm in the environment.
    
        Parameters
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        config : dict
            File consisting of simulation config.
        """
        ee_index = 11
        # The following may be modified but should not be too small of a value as test will not pass (default is 0.1)
        acceptable_proximity = 0.1
        env_instance.reset()
        for i in range(env_instance.num_arms):
            target_ee_position, _ = shuffle_arm_position(config, env_instance.robots[i], 'arm-'+str(i))
            final_position = env_instance.bullet_client.getLinkState(env_instance.robots[i].arm_id, ee_index)[0]
            difference = [abs(final_position[j] - target_ee_position[j]) for j in range(len(target_ee_position))]
            assert np.array([proximity < acceptable_proximity for proximity in difference]).all()
            env_instance.reset()
    
    def test_reset(self, env_instance, config, num_franka_joints):
        """
        Tests the reset method of FrankaArm. For each arm in the environment, shuffles the arms position then resets it and
        ensures that its actual position (after being reset) is the same as the default arm position specified in the
        config file.
    
        Parameters
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        config : dict
            File consisting of simulation config.
        num_franka_joints : int
            Fixed number of joints in Franka arm.
        """
        env_instance.reset()
        for i in range(env_instance.num_arms):
            shuffle_arm_position(config, env_instance.robots[i], 'arm-'+str(i))
            env_instance.robots[i].reset()
            arm_id = env_instance.robots[i].arm_id
            actual_pose = [env_instance.bullet_client.getJointState(arm_id, k)[0] for k in range(num_franka_joints)]
            assert actual_pose == read_default_joint_positions_from_config(config, 'arm-' + str(i), num_franka_joints)
        env_instance.reset()


# HELPER FUNCTIONS
def shuffle_arm_position(configuration_dict, arm, arm_name):
    """
    Moves the end effector of a Franka arm to a random position relative to the base of the arm. The offset from the
    base is in the range [(0.1-0.7), (0.1-0.7), 0.15]. Returns a tuple consisting of the target end effector position
    and the joint poses as calculated by inverse kinematics.

    Parameters
    ----------
    config : dict
        File consisting of simulation config.
    arm : gym_env.components.franka_arm.FrankaArm
        Arm to move.
    arm_name : string
        Name of the arm to move in the form 'arm-#'
    Returns
    -------
    tuple
        Two numpy arrarys: target_ee_position is the target end effector position, length 3 and joint_poses are the
        target pose of each joint as calculated by inverse kinematics such that the end effector is in the target
        position, length 9.
    """
    x_offset, y_offset, z_offset = random.uniform(0.1, 0.7), random.uniform(0.1, 0.7), 0.15
    base_position = configuration_dict['arms'][arm_name]['position']
    target_ee_position = np.array(base_position) + np.array([x_offset, y_offset, z_offset])
    upright_orientation = [0.718887431915408, -0.6933274427520395, -0.03597435521934313, -0.03469528970248905]
    joint_poses = arm.get_joint_poses_ik(target_ee_position=target_ee_position,
                                         target_ee_orientation=upright_orientation,
                                         use_null_space=True)
    arm.apply_action_pos(joint_poses)
    return target_ee_position, joint_poses


def shuffle_arm_velocity(arm):
    """
    Applies a velocity to each joint in a Franka arm. Returns the target velocity of each joint.

    Parameters
    ----------
    arm : gym_env.components.franka_arm.FrankaArm
        Arm to move.
    Returns
    -------
    list
        Target velocity of each joint, length 9.
    """
    target_vel = arm.max_velocities
    arm.apply_action_vel(target_vel)
    return target_vel


def read_default_joint_positions_from_config(configuration_dict, arm_name, num_franka_joints):
    """
    Reads the default joint poses from the dictionary containing the config file data and returns them as a list.

    Parameters
    ----------
    config : dict
        File consisting of simulation config.
    arm_name : string
        Name of the arm to move in the form 'arm-#'.
    num_franka_joints : int
        Fixed number of joints in Franka arm.
    Returns
    -------
    list 
        Default joint poses as specified in the config file as a list, length 12.
    """
    return [configuration_dict['arms'][arm_name]['joints']['joint-' + str(j)] for j in
            range(num_franka_joints)]
