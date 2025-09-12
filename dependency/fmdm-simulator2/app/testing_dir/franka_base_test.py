import random
import pytest
import numpy as np
from math import radians


@pytest.mark.base_functionality
class TestFrankaBaseEnv:
    def test_set_default_arm_configs(self, env_instance, config):
        """
        Tests the set_default_arm_configs method of FrankaBaseEnv. For each arm in the environment, checks if the base of
        the arm is in the position (accounts for frame offset) and orientation specified in the config file. The assert
        statements are done in the subtest_arm_base_pos_orn function.

        Parameters
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        config : dict
            File consisting of simulation config.
        """
        for i in range(env_instance.num_arms):
            arm_name = 'arm-' + str(i)
            frame_offset = 0.05
            default_base_position = config['arms'][arm_name]['position']
            default_base_position[2] = default_base_position[2] + frame_offset
            subtest_arm_base_pos_orn(env_instance=env_instance,
                                     default_base_position=default_base_position,
                                     default_base_orientation=config['arms'][arm_name]['orientation'],
                                     actual_base_info=env_instance.bullet_client.getBasePositionAndOrientation(
                                         env_instance.robots[i].arm_id))

    def test_check_for_collision_with_bin(self, env_instance, config):
        """
        Tests the check_for_collision_with_bin method of FrankaBaseEnv. Resets the environment and then checks the config
        file to see if the terminate-on-collision-with-bin condition is enabled and then for each arm in the environment
        moves it so that it comes into contact with a bin. Then calls check_for_collision_with_bin and depending on whether
        the termination condition is enabled, asserts the return values accordingly.

        Parameters
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        config : dict
            File consisting of simulation config.
        """
        env_instance.reset()
        terminate_on_collision_with_bin = config['termination']['terminate-on-collision-with-bin']['enable']
        for robot in env_instance.robots:
            assert env_instance.check_for_collision_with_bin(robot.arm_id, terminate_on_collision_with_bin) is False
            change_position_of_arm(robot, config['scene']['bin']['position'])
            assert env_instance.check_for_collision_with_bin(robot.arm_id, terminate_on_collision_with_bin) is \
                    (True if terminate_on_collision_with_bin else False)
            env_instance.reset()

    def test_check_termination_on_collision_with_bin(self, env_instance, config):
        """
        Tests the check termination on collision with bin aspect of the check_termination method of FrankaBaseEnv. Resets
        the environment and then checks the config file to see if the terminate-on-collision-with-bin condition is enabled
        and then for each arm in the environment moves it so that it comes into contact with a bin. Then calls
        check_termination and depending on whether the termination condition is enabled, asserts the return values
        accordingly.

        Parameters
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        config : dict
            File consisting of simulation config.
        """
        env_instance.reset()
        terminate_on_collision_with_bin = config['termination']['terminate-on-collision-with-bin']['enable']
        done, info = env_instance.check_termination()
        assert done is False
        for robot in env_instance.robots:
            assert info['bin-collision'] is False
            change_position_of_arm(robot, config['scene']['bin']['position'])
            done, info = env_instance.check_termination()
            assert done is (True if terminate_on_collision_with_bin else False)
            assert info['bin-collision'] is (True if terminate_on_collision_with_bin else False)
            env_instance.reset()

    def test_get_state(self, env_instance, config):
        """
        Ensures that the rgb and depth render data that is returned as the observation both have the correct shape.

        Parameters
        ----------
        env_instance : gym_envs.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        config : dict 
            File consisting of simulation config.
        """
        height_res, width_res = config['camera']['height-resolution'], config['camera']['width-resolution']
        colors = 3
        rgb_observation, depth_observation = env_instance.get_state()
        assert rgb_observation.shape == (height_res, width_res, colors)
        assert depth_observation.shape == (height_res, width_res)
        correct_rgb_observation, correct_depth_observation = env_instance.render(mode='rgb_and_depth_arrays')
        assert len(rgb_observation) == len(correct_rgb_observation)
        assert len(depth_observation) == len(correct_depth_observation)
        assert np.array([rgb_observation[i] == correct_rgb_observation[i] for i in range(len(rgb_observation))]).all()
        assert np.array([depth_observation[i] == correct_depth_observation[i] for i in
                         range(len(depth_observation))]).all()

    def test_get_reward(self, env_instance):
        """
        Ensures that the proper rewards are given when certain conditions are true or false.
        
        Parameters
        ----------
        env_instance : gym_envs.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        """
        info = {'bin-collision': False, 'grab-limit-exceeded': False, 'successful-grab': False}
        reward_when_all_conditions_false = -0.1
        reward_when_bin_collision_true = -1
        reward_when_successful_grab_true = 1
        assert env_instance.get_reward(info) == reward_when_all_conditions_false
        info['successful-grab'] = True
        assert env_instance.get_reward(info) == reward_when_successful_grab_true
        info['bin-collision'] = True
        assert env_instance.get_reward(info) == reward_when_bin_collision_true

    def test_reset(self, env_instance, config):
        """
        Tests the reset method of FrankaBaseEnv. Shuffles all the environment attributes (that are set to zero upon
        FrankaBaseEnv.reset() being called) to non-zero values, and then moves all arms to random positions. Reset
        is then called and all values within the scope of reset are checked to ensure that they have been set back
        to their default values.

        Parameters
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        config : dict
            File consisting of simulation config.
        """
        env_instance.reset()
        shuffle_env_attributes(env_instance)
        ee_cartesian_dof = 3
        num_franka_joints = 12
        for i in range(env_instance.num_arms):
            target_position = config['arms']['arm-'+str(i)]['position']
            target_position = [target_position[j] + random.uniform(0.15, 0.6) for j in range(ee_cartesian_dof)]
            change_position_of_arm(env_instance.robots[i], target_position)
        env_instance.reset()
        assert env_instance.reward == 0
        assert env_instance.grabbed_obj_ids == []
        for j in range(env_instance.num_arms):
            config_default_joint_pose = [config['arms']['arm-'+str(j)]['joints']['joint-' + str(k)] for k in
                                         range(num_franka_joints)]
            actual_pose = [env_instance.bullet_client.getJointState(env_instance.robots[j].arm_id, k)[0] for k in
                           range(num_franka_joints)]
            assert actual_pose == config_default_joint_pose
        original_obj_type = env_instance.scene.bin_with_objs.path_to_current_object
        env_instance.reset(keep_object_type=True)
        new_obj_type = env_instance.scene.bin_with_objs.path_to_current_object
        assert original_obj_type == new_obj_type

    def test_remove_single_object_from_bin(self, env_instance):           
        """                                                          
        Tests the remove_single_object_from_bin method of FrankaBaseEnv. Ensures that an object is removed from the
        pybullet simulation and the change in number of objects is properly tracked by BinScene and BinPicking.                                        
        
        Parameters                                              
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        """
        initial_total_number_of_bodies = env_instance.bullet_client.getNumBodies()
        initial_number_of_objects = env_instance.get_current_number_of_objects()
        random_object = random.choice(env_instance.scene.bin_with_objs.objects)
        env_instance.remove_single_object_from_bin(random_object)
        final_total_number_of_bodies = env_instance.bullet_client.getNumBodies()
        final_number_of_objects = env_instance.get_current_number_of_objects()
        assert final_total_number_of_bodies == initial_total_number_of_bodies - 1
        assert final_number_of_objects == initial_number_of_objects - 1
        assert random_object not in env_instance.scene.bin_with_objs.objects
        
    def test_remove_objects_not_in_bins(self, env_instance):
        """
        Tests the remove_objects_not_in_bins method of FrankaBaseEnv. Calls the method and then ensures that all remaining
        objects are in the bin.

        Parameters
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        """
        initial_total_number_of_bodies = env_instance.bullet_client.getNumBodies()
        initial_number_of_objects = env_instance.get_current_number_of_objects()
        env_instance.remove_objects_not_in_bins()
        position = 0
        for obj in env_instance.scene.bin_with_objs.objects:
            obj_position = env_instance.bullet_client.getBasePositionAndOrientation(obj)[position]
            assert env_instance.scene.bin_with_objs.contains(obj_position)
        final_total_number_of_bodies = env_instance.bullet_client.getNumBodies()
        final_number_of_objects = env_instance.get_current_number_of_objects()
        assert (final_total_number_of_bodies - initial_total_number_of_bodies == final_number_of_objects -
                initial_number_of_objects)

    def test_get_current_number_of_objects(self, env_instance):
        """
        Tests the get_current_number_of_objects method of FrankaBaseEnv. Ensures that the method is returning the correct
        data.

        Parameters
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        """
        assert env_instance.get_current_number_of_objects() == env_instance.scene.current_number_of_objects


# HELPER FUNCTIONS
def subtest_arm_base_pos_orn(env_instance, default_base_position, default_base_orientation, actual_base_info):
    """
    Given the default position and orientation (in euler and degree form) of the base of an arm, compares the values to
    the actual position and orientation of the base of the arm in the environment and ensures that they are the same.

    Parameters
    ----------
    env_instance : gym_env.envs.franka_v#.Franka#Env
        An instance of a franka gym env.
    default_base_position : list
        The default base position of the arm as specified in the config file (accounts for frame offset, z + 0.05), length 3
    default_base_orientation : list
        The default orientation of the arm as specified in the config file (in euler and degree form), length 3
    actual_base_info : tuple
        Consisting of the position as a list of 3 floats as [x, y, z] and orientation as a list of 4 floats as [x, y, z, w]
    """
    orn_in_radians = [radians(default_base_orientation[i]) for i in range(len(default_base_orientation))]
    orn_in_quaternion = env_instance.bullet_client.getQuaternionFromEuler(orn_in_radians)
    assert list(actual_base_info[0]) == default_base_position
    assert actual_base_info[1] == orn_in_quaternion


def change_position_of_arm(arm, target_position):
    """
    Given a target end effector position and an arm to apply an action to, this function uses inverse kinematics to
    calculate the joint poses such that the end effector of the arm is in the specified position.

    Parameters
    ----------
    arm : gyme_env.components.franka_arm.FrankaArm
        Arm to apply action to.
    target_position : list
        Target end effector position, length 3.
    """
    upright_orientation = [0.718887431915408, -0.6933274427520395, -0.03597435521934313, -0.03469528970248905]
    target = arm.get_joint_poses_ik(target_ee_position=target_position,
                                    target_ee_orientation=upright_orientation,
                                    use_null_space=True)
    arm.apply_action_pos(target)


def shuffle_env_attributes(env_instance):
    """
    Changes the env attributes that are reset when env_instance.reset() is called.

    Parameters
    ----------
    env_instance : gym_env.envs.franka_v#.Franka#Env
        An instance of a franka gym env.
    """
    env_instance.reward = 3
    env_instance.grabbed_obj_ids = [1, 2, 3]
