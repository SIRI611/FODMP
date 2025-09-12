import os
import numpy as np
from gym import spaces
from gym_env.envs import base_env
from gym_env.utils import bin_utils
from gym_env.utils import scene_utils
from gym_env.utils import robot_utils
from gym_env.utils import object_utils


class FrankaEnv0(base_env.BaseGymEnv):
    """
    Franka gym environment for one arm, positional action space, and observation space consisting of the rgb
    and depth camera render data.
    """
    def __init__(self, config_file):
        """
        Initializes a franka gym environment for one arm and the space attributes as specified in
        set_space_attributes(). Sets up parameters required to load the scene - reset() must be called to load the
        environment.

        Parameters
        ----------
        config_file : str
            Path to yaml file consisting of simulation configuration.
        """
        super().__init__(config_file)
        self.plane_id, self.drop_bin_id, self.bin_with_objs_id, self.objects = None, None, None, None
        self.franka_robot = None
        self.grabs_before_termination, self.successful_grabs = 0, 0
        assert self.render_mode == 'rgb_and_depth_arrays', \
            'This environment\'s observation space consists of both the rgb and depth data so the render mode must ' \
            'be \'rgb_and_depth_arrays\'.'
        #self.set_space_attributes()

    def check_termination(self):
        """
        Checks whether or not the simulation has reached a terminating state. In this environment a terminating state is
        either when when an arm collides with a bin (as implemented below, this function checks if there are any contact
        points between any arm and any bin) or when the simulation has exceeded the specified number of successful
        grabs. These terminating states can be enabled/disabled and certain aspects of them can be adjusted in the
        config file. All termination info is kept track of in a dictionary and returned for reward calculation purposes.

        Returns
        -------
        tuple
            A boolean value and a dictionary. The boolean value represents whether or not the simulation is in a
            terminating state and the dictionary contains auxillary diagnostic info including a boolean value regarding
            whether any arm has collided with a bin, whether the grab limit has been exceeded, and the number of
            completed grabs.
        """
        info = {'bin-collision': False, 'grab-limit-exceeded': False, 'successful-grab': False}
        if self.config_file['termination']['terminate-on-collision-with-bin']['enable']:
            info['bin-collision'] = robot_utils.check_collision_between_robot_and_object(
                self.bullet_client,
                self.franka_robot.robot_id,
                self.bin_with_objs_id,
            )
        if self.config_file['termination']['terminate-after-successful-grabs']['enable']:
            bin_position, _ = self.bullet_client.getBasePositionAndOrientation(self.bin_with_objs_id)
            bin_height = self.config_file['scene']['bin']['dimensions'][2]
            minimum_grasp_height = bin_position[2] + (1.5 * bin_height)
            info['successful-grab'], info['grasped-object-id'] = robot_utils.check_grasp(
                self.bullet_client,
                self.franka_robot,
                self.objects,
                minimum_grasp_height,
                self.config_file['termination']['terminate-after-successful-grabs']['ensure-grab-is-stable']
            )
            if info['successful-grab']:
                self.successful_grabs += 1
            info['grab-limit-exceeded'] = self.successful_grabs >= self.grabs_before_termination
        return info['bin-collision'] or info['grab-limit-exceeded'], info

    def get_reward(self, info):
        """
        Calculates the reward based on the current state of the agent and the environment.

        Parameters
        ----------
        info : dict

        Returns
        -------
        float
            Value of the reward.
        """
        if info['bin-collision']:
            return -1
        if info['successful-grab']:
            return 1
        return -0.1

    def get_state(self):
        """
        Gets the rgb and depth camera render data as a representation of the current state of the env.

        Returns
        -------
        tuple
            Two numpy arrays that correspond to the rgb and depth camera data respectively. RGB array has shape
            (height x width x 3) where the 3 corresponds to the number of colors. Depth array has shape (height x
            width). In general terms it is an observation of the current state of the franka arm in the environment
            (same format as described by this envs observation_space).
        """
        return self.render(self.render_mode)

    def initialize_scene(self):
        """
        Initializes all the aspects of the scene. Loads the plane, drop-bin (if enabled in config file), bin, objects,
        and franka robot and sets them all as class attributes. The finger indices and grasptarget index passed to the
        picking robot constructor are as defined in the panda.urdf file (urdf representation of the robot).
        """
        path_to_plane_urdf = os.environ['path-to-assets'] + 'plane/plane.urdf'
        self.plane_id = scene_utils.load_plane(self.bullet_client, path_to_plane_urdf)
        if self.config_file['scene']['drop-bin-enable']:
            self.drop_bin_id, _ = bin_utils.load_bin(
                self.bullet_client,
                self.config_file['scene']['drop-bin'],
                self.ml_mode
            )
        self.bin_with_objs_id, self.objects = bin_utils.load_bin(
            self.bullet_client,
            self.config_file['scene']['bin'],
            self.ml_mode
        )
        object_utils.allow_objects_to_fall(self.bullet_client, self.config_file['scene']['steps-for-objects-to-fall'])
        self.franka_robot = robot_utils.load_robot(
            bullet_client=self.bullet_client,
            robot_config=self.config_file['robot'],
            finger_indices=[9, 10],
            grasptarget_index=11
        )

    def reset(self):
        """
        Resets the simulation and class-relevant environment attributes to their default settings. Then reloads the
        scene. Possible to reload the scene with the same type of objects that were in the bin prior to calling reset.

        Returns
        -------
        object
            Observation of the current state of this env in the format described by this envs observation_space.
        """
        super().reset()
        self.initialize_scene()
        gbt = self.config_file['termination']['terminate-after-successful-grabs']['grabs-before-termination']
        self.grabs_before_termination = len(self.objects) if len(self.objects) < gbt else gbt
        if self.sim_mode == 'gpu-gui':
            scene_utils.enable_rendering_in_gui(self.bullet_client, self.config_file['simulation']['enable-windows'])
        self.set_space_attributes()
        return self.get_state()

    def set_space_attributes(self):
        """
        Defines the action space, observation space, and reward range. The action space supports positional control for
        a franka arm (9 dimension box with respective lower and upper positional bounds for each joint or end
        effector). Observation space consists of the rgb and depth camera render data.
        """
        self.action_space = spaces.Box(
            low=np.array(self.franka_robot.lower_limits),
            high=np.array(self.franka_robot.upper_limits),
            dtype=np.float32
        )
        self.observation_space = spaces.Tuple((
            spaces.Box(
                low=0,
                high=255,
                shape=(
                    self.config_file['camera']['height-resolution'],
                    self.config_file['camera']['width-resolution'],
                    3
                ),
                dtype=np.uint8
            ), spaces.Box(
                low=0,
                high=255,
                shape=(
                    self.config_file['camera']['height-resolution'],
                    self.config_file['camera']['width-resolution']
                ),
                dtype=np.uint8
            )
        ))
        self.reward_range = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def step(self, action):
        """
        Runs one step of the simulation. Applies a position control action to the franka arm and gets an observation.

        Parameters
        ----------
        action : object
            The action to apply to the model

        Returns
        -------
        tuple
            Observation (object), reward (float), done (bool), info (dict). Where observation represents
            the current state of the environment, reward is the amount of reward for the previous action, done is
            whether or not to terminate the simulation (based on termination config), and info contains auxillary
            diagnostic info (useful for debugging and sometimes training).
        """
        self.franka_robot.apply_action_pos(action)
        observation = self.get_state()
        done, info = self.check_termination()
        reward = self.get_reward(info)
        return observation, reward, done, info
