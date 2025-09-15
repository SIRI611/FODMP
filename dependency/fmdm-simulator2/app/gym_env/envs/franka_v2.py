import os
import time
import numpy as np
from gym import spaces
from PIL import Image
from transforms3d.euler import euler2mat
from gym_env.envs import base_env
from gym_env.utils import bin_utils
from gym_env.utils import scene_utils
from gym_env.utils import robot_utils
from gym_env.utils import object_utils


class FrankaEnv2(base_env.BaseGymEnv):
    """
    Franka gym environment for one arm. The action space consists of the desired change in position of the Franka
    robot's gripper, the desired change in angle of the Franka robot's gripper, and whether to open or close the
    gripper. The observation space consists of the current position of the Franka robot's gripper in world
    coordinates, an array of rgb image data (visual representation of the environment), and whether the gripper
    is open or closed (1 if open, 0 if closed).
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
        super().__init__(config_file=config_file)
        self.plane_id, self.drop_bin_id, self.bin_with_objs_id, self.objects = None, None, None, None
        self.franka_robot = None
        self.timeout, self.grasp_success, self._end_episode, self.grasped_object = False, False, False, False
        self.env_step, self.resets = 0, 0
        self.max_steps, self.resets_per_setup = self.config_file['env']['max-steps'], self.config_file['env']['resets-per-setup']
        self.da, self.dv = self.config_file['env']['da'], self.config_file['env']['dv']
        self.downsample_height = self.config_file['env']['downsample-height']
        self.downsample_width = self.config_file['env']['downsample-width']
        self.bullet_client.resetDebugVisualizerCamera(0.5, 40, -40, [0, 0, 0])        
        assert self.render_mode == 'rgb_and_depth_arrays', \
            'This environment is structured based on the assumption that the render mode is \'rgb_and_depth_arrays\'.'
        self.initialize_environment()

    def check_termination(self):
        """
        Checks whether the current episode is in a state that satisfies a terminating condition.

        Returns
        -------
        bool
            True if the current episode is in a state that satisfies a terminating condition. False otherwise. 
        """
        if self.env_step >= self.max_steps:
            self.timeout = True
        if self.grasped_object and self.end_episode:
            print(self.grasped_object)
            object_utils.remove_object(self.bullet_client, self.grasped_object)
            self.object_ids.remove(self.grasped_object)
            self.grasped_object = None
        return (self.timeout or self.end_episode)

    def draw_debug_frame(self, position, orientation, size=0.2):
        """
        Draws a debug frame around the robot's grasptarget.

        Parameters
        ----------
        position : 
            Current position of the robot's grasptarget.
        orientation :
            Current orientation of the robot's gripper.
        size : float
            Size of debug frame. 
        """
        self.bullet_client.addUserDebugLine(position, position + size * orientation[:, 0], [1, 0, 0], 2)
        self.bullet_client.addUserDebugLine(position, position + size * orientation[:, 1], [0, 1, 0], 2)
        self.bullet_client.addUserDebugLine(position, position + size * orientation[:, 2], [0, 0, 1], 2)

    def get_image_observation(self):
        """
        Gets an array consisting of a rgb image representation of the current state of the environment and 
        then downsamples the image to the specified resolution.

        Returns
        -------
        numpy.ndarray
            Array consisting of the downsampled rgb image data. Shape is (downsample_height, downsample_width).
        """
        rgb_array, depth_array = self.render(self.render_mode)
        rgb_image = Image.fromarray(rgb_array)
        downsampled_rgb_image = rgb_image.resize((self.downsample_height, self.downsample_width), Image.ANTIALIAS)
        downsampled_rgb_array = np.array(downsampled_rgb_image)
        return downsampled_rgb_array

    def get_info(self):
        """
        Gets auxillary info corresponding to the current state of the environment.

        Returns
        -------
        dict
            Consists of extra info corresponding to the current state of the environment.
        """           
        info = {
            'grasp_success': self.grasp_success,
            'timeout': self.timeout
        }
        return info

    def get_reward(self):
        """
        Checks if the franka robot has completed a successful grab and returns an award accordingly.

        Returns
        -------
        int
            The value of the reward to give based on the current state of the environment.
        """
        self.grasp_success, self.grasped_object = robot_utils.check_grasp(
            self.bullet_client,
            self.franka_robot,
            self.object_ids,
            self.config_file['env']['success-height'],
            self.config_file['termination']['terminate-after-successful-grabs']['ensure-grab-is-stable']
        )
        return int(self.grasp_success and self.end_episode)

    def get_state(self):
        """
        Returns an observation of the environment as described by this env's observation space. Specifically
        the current position of the Franka robot's gripper in world coordinates, an array of rgb image data
        (visual representation of the environment), and whether the gripper is open or closed (1 if open,
        0 if closed).

        Returns
        -------
        dict
            Information pertaining to the grasptarget position, an array of rgb image data, and the status
            of the gripper.
        """
        robot_utils.mount_camera_on_robot(self.franka_robot, self.camera)
        grasptarget_pos, gripper_orn = self.franka_robot.get_grasptarget_state()
        gripper_open = -1 if self.franka_robot.gripper_is_open else 1
        observation = {
            'grasptarget': np.concatenate((
                grasptarget_pos, 
                [np.sin(gripper_orn[-1]), np.cos(gripper_orn[-1])],
                [gripper_open]
            )).astype(np.float32),
            'image': self.get_image_observation(),
        }
        return observation

    def initialize_environment(self):
        """
        Respawns the environment and resets the action and observation spaces as well as the reward range and then
        reinitializes the scene.
        """
        super().reset()
        self.initialize_scene()
        if self.sim_mode == 'gpu-gui':
            scene_utils.enable_rendering_in_gui(self.bullet_client, self.config_file['simulation']['enable-windows'])
            self.bullet_client.configureDebugVisualizer(self.bullet_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        self.set_space_attributes()

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
        self.bin_with_objs_id, self.object_ids = bin_utils.load_bin(
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

        Parameters
        ----------
        keep_object_type : bool
            Whether or not the scene should be reloaded with the same object type in each bin or a different object
            type.
        Returns
        -------
        object
            Observation of the current state of this env as described by this envs observation_space.
        """
        self.resets += 1
        self.franka_robot.reset()
        self.franka_robot.open_gripper()
        for _ in range(5):
            robot_utils.apply_delta_action(
                self.bullet_client,
                self.franka_robot,
                [0, 0, -self.dv, 0, -1],
                self.config_file['env']['workspace']['upper-bound'],
                self.config_file['env']['workspace']['lower-bound']
            )
        grasptarget_pos, gripper_orn = self.franka_robot.get_grasptarget_state()
        self.draw_debug_frame(grasptarget_pos, euler2mat(*gripper_orn))
        self.bullet_client.removeAllUserDebugItems()
        if (len(self.object_ids) == 0) or (self.resets % self.resets_per_setup == 0):
            self.initialize_environment()
        self.timeout, self.grasp_success, self._end_episode, self.grasped_object = False, False, False, False
        self.env_step = 0
        return self.get_state(), self.get_info()

    def set_space_attributes(self):
        """
        Defines the action space, observation space, and reward range in terms of the gym.spaces module. The action
        space is an array with shape (6,) and corresponds to the following commands [dx,dy,dz,da,open/close gripper, 
        end episode] where dx, dy, and dz are the desired change in position of the franka robot's end effector, da is
        the desired change in top-down gripper angle delta, and the last element is a 1 or a -1 in order to open or
        close the gripper, respectively. The observation space is a dictionary consisting of the current position
        of the franka robot's gripper, an array consisting of the rgb image data (visual representation of the
        environment), and whether the gripper is open or closed.
        """
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'gripper_position': spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(self.downsample_height, self.downsample_width, 3),
                dtype=np.uint8
            ),
        })
        self.reward_range = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    
    def step(self, action):
        """
        Runs one step of the simulation. Applies a position control action to the franka arm and gets a
        position/velocity observation.

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
        assert self.action_space.contains(action), \
            'Action is not formatted correctly, please refer to the action space description for info on the correct' \
            'format of an action for this environment.'
        action[:3] = [self.dv * delta_position for delta_position in action[:3]]
        action[3] *= self.da
        rxn_forces = robot_utils.apply_delta_action(
            self.bullet_client,
            self.franka_robot,
            action[:5],
            self.config_file['env']['workspace']['upper-bound'],
            self.config_file['env']['workspace']['lower-bound']
        ) 
        self.end_episode = True if action[-1] > 0 else False
        self.env_step += 1
        observation = self.get_state()
        reward = self.get_reward()
        done = self.check_termination()
        info = self.get_info()
        return observation, reward, done, info
