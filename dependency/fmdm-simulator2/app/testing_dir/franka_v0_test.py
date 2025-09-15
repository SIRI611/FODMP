import pytest
import numpy as np
from gym_env.envs.franka_v0 import Franka0Env


@pytest.mark.franka_v0
class TestFranka0Env:
    def test_set_space_attributes(self, env_instance, config):
        """
        Tests the set_space_attributes method of Franka0Env. Ensures that the current env instance is an instance of
        Franka0Env and if so, asserts that the number of arms in the env are correct and that the observation and action
        space as well as the reward range, all have the correct shape.

        Parameters
        ----------
        env_instance : gym_envs.envs.franka_v0.Franka0Env
            An instance of a franka gym env.
        config : dict 
            File consisting of simulation config.
        """
        if not isinstance(env_instance, Franka0Env):
            pytest.skip('Working environment is not an instance of Franka0Env')
        moveable_joints = 9
        assert env_instance.num_arms == 1
        assert env_instance.action_space.shape == (moveable_joints,)
        height_res, width_res = config['camera']['height-resolution'], config['camera']['width-resolution']
        rgb_data_index, depth_data_index, correct_observation_size, colors = 0, 1, 2, 3
        assert len(env_instance.observation_space) == correct_observation_size
        assert env_instance.observation_space[rgb_data_index].shape == (height_res, width_res, colors)
        assert env_instance.observation_space[depth_data_index].shape == (height_res, width_res)
        assert env_instance.reward_range.shape == (1,)
        assert env_instance.reward_range.is_bounded(manner='both')
        assert env_instance.reward_range.contains([-1]) and env_instance.reward_range.contains([1])

    def test_reset(self, env_instance, config):
        """
        Tests the reset method of Franka0Env. Ensures that all the space attributes are set properly and that the
        returned observation has the correct shape.

        Parameters
        ----------
        env_instance : gym_envs.envs.franka_v0.Franka0Env
            An instance of a franka gym env.
        config : dict 
            File consisting of simulation config.
        """
        if not isinstance(env_instance, Franka0Env):
            pytest.skip('Working environment is not an instance of Franka0Env')
        observation = env_instance.reset()
        assert env_instance.action_space.shape == (9,)    
        height_res, width_res = config['camera']['height-resolution'], config['camera']['width-resolution']
        rgb_data_index, depth_data_index, correct_observation_size, colors = 0, 1, 2, 3
        assert len(env_instance.observation_space) == correct_observation_size
        assert env_instance.observation_space[rgb_data_index].shape == (height_res, width_res, colors)
        assert env_instance.observation_space[depth_data_index].shape == (height_res, width_res)
        assert env_instance.reward_range.shape == (1,) 
        assert len(observation) == correct_observation_size
        assert observation[rgb_data_index].shape == (height_res, width_res, colors)
        assert observation[depth_data_index].shape == (height_res, width_res)
        original_obj_type = env_instance.scene.bin_with_objs.path_to_current_object
        env_instance.reset(keep_object_type=True)
        new_obj_type = env_instance.scene.bin_with_objs.path_to_current_object
        assert original_obj_type == new_obj_type

    def test_step(self, env_instance, config):
        """
        Tests the step method of Franka0Env. Ensures that the current env instance is an instance of Franka0Env and if
        so, checks to make sure that the correct variables are being adjusted within the step method and that the
        correct values are being returned.

        Parameters
        ----------
        env_instance : gym_envs.envs.franka_v0.Franka0Env
            An instance of a franka gym env.
        config : dict 
            File consisting of simulation config.
        """
        if not isinstance(env_instance, Franka0Env):
            pytest.skip('Working environment is not an instance of Franka0Env')
        types = [object, float, bool, dict]
        action = env_instance.action_space.sample()
        data = env_instance.step(action)
        observation, reward, done, info = data
        assert np.array([type(data[i] is types[i] for i in range(len(data)))]).all() 
        height_res, width_res = config['camera']['height-resolution'], config['camera']['width-resolution']
        rgb_data_index, depth_data_index, correct_observation_size, colors = 0, 1, 2, 3
        assert len(observation) == correct_observation_size
        assert observation[rgb_data_index].shape == (height_res, width_res, colors)
        assert observation[depth_data_index].shape == (height_res, width_res)
        correct_observation = env_instance.render(mode='rgb_and_depth_arrays')
        assert np.array([observation[rgb_data_index][i] == correct_observation[rgb_data_index][i] for i in
                         range(len(observation[rgb_data_index]))]).all()
        assert np.array([observation[depth_data_index][i] == correct_observation[depth_data_index][i] for i in
                         range(len(observation[depth_data_index]))]).all()
        assert (done, info) == env_instance.check_termination()
        assert reward == env_instance.get_reward(info)
