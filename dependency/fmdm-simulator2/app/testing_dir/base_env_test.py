import pytest
import random
import re


@pytest.mark.base_functionality
class TestBaseGymEnv:
    """
    NOTE the reset method is tested by FrankaBaseEnv.reset() and the close method is tested in conftest.py after all tests
    are run.
    """
    def test_configure(self, env_instance, config):
        """
        Tests the configure method of BaseGymEnv.
        
        Parameters
        ----------
        env_instance : gym_env.envs.*
            An instance of a gym env.
        config : dict
            File consisting of simulation config.
        """
        assert type(env_instance.config_file) is dict
        assert env_instance.config_file == config

    def test_initialize_client(self, env_instance):
        """
        Tests the initialize_client method of BaseGymEnv. If a valid connection to the physics engine has been established,
        the environment will have a non-negative physics client id.

        Parameters
        ----------
        env_instance : gym_env.envs.*
            An instance of a gym env.
        """
        assert env_instance.physics_client_id > -1

    def test_load_scene(self, env_instance, config):
        """
        Tests the load_scene method of BaseGymEnv. Ensures that the scene that the environment has loaded is the same as
        the scene specified in the configuration file.
        
        Parameters
        ----------
        env_instance : gym_env.envs.*
            An instance of a gym env.
        config : dict
            File consisting of simulation config
        """
        scene_class = 1
        mode_from_config = config['scene']['mode']
        loaded_scene = re.sub(r'\'>', '', re.sub(r'<[a-z]+ \'([a-zA-Z_]+\.)*', '', str(type(env_instance.scene))))
        expected_scene = env_instance.scene_modes[mode_from_config][scene_class]
        assert loaded_scene == expected_scene

    def test_render(self, env_instance, config):
        """
        Tests the render method of BaseGymEnv. Ensures that nothing is returned when human mode is specified, and that
        the shape of the rgb array is correct when the rgb_array mode is specified. This also inherently ensures that
        when rgb_array mode is specified and get_depth_data is not set, no depth data is returned (i.e. get_depth_data
        defaults to False). Lastly tests that when get_depth_data is set to True and rgb_array mode is specified,
        both the rgb and depth arrays have the correct shape.

        Parameters
        ----------
        env_instance : gym_env.envs.*
            An instance of a gym env.
        config : dict
            File consisting of simulation config.
        """
        none_value = env_instance.render(mode='human')
        assert none_value is None
        width_res = config['camera']['width-resolution']
        height_res = config['camera']['height-resolution']
        color_rgb_data = env_instance.render(mode='rgb_array')
        num_colors = 3
        assert color_rgb_data.shape == (height_res, width_res, num_colors)
        color_rgb_data, depth_data = env_instance.render(mode='rgb_and_depth_arrays')
        assert color_rgb_data.shape == (height_res, width_res, num_colors)
        assert depth_data.shape == (height_res, width_res)

    def test_recompute_view_and_projection_matrix(self, env_instance, config):
        """
        Tests the recompute_view_and_projection_matrix method of BaseGymEnv. Uses values that are known to work
        (original default.yaml camera configuration). Calculates the expected view and projection matrices directly using
        the pybullet API and ensures that those computed by the method in BaseGymEnv are the same.
        
        Parameters
        ----------
        env_instance : gym_env.envs.*
            An instance of a gym env.
        config : dict
            File consisting of simulation config.
        """
        env_instance.camera.view_matrix, env_instance.camera.projection_matrix = None, None
        eye_position, target_position, up_vector = [0, 0, 0.25], [0, 0, 0], [0, 1, 0]
        fov, width_res, height_res, near_plane, far_plane = 76, 480, 480, 0.008, 0.16
        correct_view_matrix = env_instance.bullet_client.computeViewMatrix(
            cameraEyePosition=eye_position,
            cameraTargetPosition=target_position,
            cameraUpVector=up_vector
        )
        correct_projection_matrix = env_instance.bullet_client.computeProjectionMatrixFOV(
            fov=fov,
            aspect=width_res / height_res,
            nearVal=near_plane,
            farVal=far_plane
        )
        env_instance.recompute_view_and_projection_matrix(
            eye_position,
            target_position,
            fov,
            near_plane,
            far_plane,
            up_vector,
            width_res,
            height_res
        )
        assert env_instance.camera.view_matrix == correct_view_matrix
        assert env_instance.camera.projection_matrix == correct_projection_matrix

    def test_seed(self, env_instance):
        """
        Tests the seed method of BaseGymEnv. Ensures that the seed value that is passed to the method is the same value
        that is set as the seed of the env and returned as the sole element of a list.
    
        Parameters
        ----------
        env_instance : gym_env.envs.*
            An instance of a gym env.
        """
        desired_seed = random.randint(1,10000)
        env_seed = env_instance.seed()[0]
        assert type(env_seed) is int 
