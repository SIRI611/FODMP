import os
import math
import time
import numpy as np
from gym import spaces
from PIL import Image
from gym_env.envs import base_env
from gym_env.utils import bin_utils
from gym_env.utils import scene_utils
from gym_env.utils import object_utils
from gym_env.components.cartesian_robot import CartesianRobot


class CartesianEnv0(base_env.BaseGymEnv):
    def __init__(self, config_file):
        """
        Initializes a gym environment for a cartesian franka robot with a positional action space
        and an observation space consisting of the rgb data and the depth data from the camera.
        
        Parameters
        ----------
        config_file : str
            Path to yaml file consisting of simulation config.
        """
        super().__init__(config_file=config_file)
        self.ml_mode, self.demo = self.config_file['simulation']['ml-mode'], self.config_file['simulation']['demo']
        self.robot_configs = self.config_file['cartesian-robot']
        self.verbose = self.robot_configs['verbose']
        self.plane_id, self.drop_bin_id, self.bin_with_objs_id, self.objects = None, None, None, None
        self.robot = None
        self.num_grabs = 0
        assert self.render_mode == 'rgb_and_depth_arrays', \
            'This environment\'s observation space consists of both the rgb and depth data so the render mode must ' \
            'be \'rgb_and_depth_arrays\'.'

    def action_calibration(self, action, collision_free=False):
        """
        Calibrate policy network output action to robot space action.
        
        Parameters
        ----------
        action : list
            [x, y, z, yaw, d]
        collision_free : bool
            Whether collision free action calibration is required.
        Returns
        -------
        tuple
            The gripper base pose in robot coordinate frame.
        """
        pos, yaw, d = action[0:3], action[3], action[4]
        d_r = (d+1) * self.robot_configs['gripper']['maximum-distance']/2
        yaw_r = yaw * np.pi / 2
        adjsx = 2*d_r*math.cos(yaw_r) if collision_free else 0
        adjsy = abs(2 * d_r * math.sin(yaw_r)) if collision_free else 0
        bin_dims = self.config_file['scene']['bin']['dimensions']
        x0, y0, z0 = self.config_file['scene']['bin']['position']
        x = (pos[0]+1)*(bin_dims[0] - adjsx)/2 - bin_dims[0]/2 + adjsx/2 + x0  # x in [-bin_dims[0]/2, bin_dims[0]/2]
        y = (pos[1]+1)*(bin_dims[1] - adjsy)/2 - bin_dims[1]/2 + adjsy/2 + y0  # y in [-bin_dims[1]/2, bin_dims[1]/2]
        z = (pos[2]+1)*(bin_dims[2])/2 + z0  # z in [0, bin_dims[2]]  0 bottom, bim_dims[2]: top
        # move to the new pose
        pos = self.robot_coordinate_to_world_cooridnate(np.asarray([x, y, z]))
        if self.verbose:
            print(' ==> calib [x:{:+.4f}, y:{:+.4f}, z:{:+.4f}, phi:{:+.2f}, d:{:+.4f}]'.format(
                pos[0],
                pos[1],
                pos[2],
                yaw_r * 180 / np.pi,
                d_r
            ))
        return pos, yaw_r, d_r

    def check_termination(self, rid=None):
        """
        Checks if a termination condition has been met.
        
        Parameters
        ----------
        rid : int or None
            Equals -1 if the cartesian robot has collided with a bin wall.
        """
        info = {
            'collide-bin': False,
            'success-pick': False,
            'obj-id': None,
            'grasped-obj-num': 0,
            'total-obj-num': len(self.objects)
        }
        if rid is -1:  # colliding with bin walls
            info['collide-bin'] = True
        else:
            obj_id = self.grasp_success()
            if obj_id is not None:  # successful grasp
                info['success-pick'], info['obj-id'] = True, obj_id
                if not self.demo:
                    object_utils.remove_object(self.bullet_client, obj_id)
                    self.objects.remove(obj_id)
                else:  # demo mode
                    self.robot.place_down()
        # update obj num status
        info['grasped-obj-num'] = self.num_grabs
        return info

    def get_observation(self):
        """
        Renders the current state of the environment. Assumes the render mode for this environment is
        'rgb_and_depth_arrays'.
        
        Returns
        -------
        tuple
            As defined by this envs observation space, the first item in the tuple is the rgb array (shape
            is height_resolution x width_resolution x 3) and the second item in the tuple is the depth array
            (shape is height_resolution x width_resolution).
        """
        self.robot.avoid_camera()
        rgb_obs, depth_obs = self.render(mode=self.render_mode)
        rgb_image = Image.fromarray(rgb_obs.astype('uint8'), 'RGB').convert('L')
        rgb_obs = np.expand_dims(np.asarray(rgb_image), axis=2)
        self.robot.back_to_initial()
        return rgb_obs, depth_obs

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
        if info['collide-bin']:
            return -1.0
        if info['success-pick']:
            return 1.0
        return -0.1

    def get_state(self):
        """
        Gets the current state of the cartesian robot.
        
        Returns
        -------
        list
            [x, y, z, yaw, d] where d is the distance between grippers.
        """
        return self.robot.get_robot_state()

    def grasp_success(self, threshold=0.03):
        """
        Decides if a grasp is successful or not.
        
        Parameters
        ----------
        threshold : float
            Any object pose above bin-top + threshold will be judged as successful grasp.
        Returns
        -------
        int or None
            Either the unique object id of the successfully grabbed object or None if no objects have been
            grabbed successfully.
        """
        for i in range(len(self.objects)):
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.objects[i])
            z0 = self.config_file['scene']['bin']['position'][2] + self.config_file['scene']['bin']['dimensions'][2]
            if list(pos)[2] >= z0 + threshold:  # 0.1 is the bin base z, 0.04 is the bin dim height
                obj_id = self.objects[i]
                object_utils.remove_object(self.bullet_client, obj_id)
                self.num_grabs += 1
                return obj_id
        return None

    def initialize_scene(self):
        """
        Initializes all the aspects of the scene. Loads the plane, drop-bin (if enabled in config file), bin, objects,
        and cartesian robot and sets them all as class attributes.
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
        self.robot = CartesianRobot(self.robot_configs, self.bullet_client, verbose=self.verbose)

    def reset(self):
        """
        Removes all objects from the simulation and resets class-relevant attributes. Then respawns the scene.
        Note: rendering is disabled during the loading process if gui mode is enabled (see BaseScene.reset()) in order
        to speed up the loading process.
        """
        super().reset()
        self.initialize_scene()
        self.num_grabs = 0
        if self.sim_mode == 'gpu-gui':
            scene_utils.enable_rendering_in_gui(self.bullet_client, self.config_file['simulation']['enable-windows'])
        self.set_space_attributes()
        return self.get_observation()

    def robot_coordinate_to_world_cooridnate(self, pos):
        """
        bc.changeConstrant controls the robot position in world coordinate. Network generates actions in
        robot coordinate, so we need to convert robot coordinate framework based actions to world coordinate
        framework based ones.
        
        Parameters
        ----------
        pos : numpy.ndarray
            Position of robot in robot coordinate framework.
        Returns
        -------
        numpy.ndarray
            Position of robot in world coordinate framework.
        """
        return pos + np.asarray(self.robot_configs["gripper"]["tip"])

    def sample_actions(self):
        """
        Gets a sample action of the same shape as this envs action space.
        
        Returns
        -------
        numpy.ndarray
            The action to apply to the model.
        """
        actions = np.random.uniform(-1, 1, 5)
        return actions

    def set_space_attributes(self):
        """
        Defines the action space, observation space, and reward range in terms of the types provided by python package
        gym.spaces (i.e. spaces.Dict(), spaces.Box()) and sets them as instance attributes. Observation space consists
        of the normalized greyscale camera image as the first item in the tuple and the greyscale depth camera image
        as the second item in the tuple.
        """
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
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
        Runs one timestep of the simulation. Applies a position control action to the cartesian robot and gets
        an observation.
        
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
        pos, yaw, d = self.action_calibration(action)
        rid = self.robot.apply_action(pos, yaw, d, self.bin_with_objs_id)
        info = self.check_termination(rid)
        obs = self.get_observation()
        reward = self.get_reward(info)
        done = info['collide-bin'] or info['success-pick']
        return obs, reward, done, info
