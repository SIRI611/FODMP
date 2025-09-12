import os
import numpy as np
import math
from gym import spaces
from gym_env.envs import base_env
from gym_env.utils import bin_utils
from gym_env.utils import scene_utils
from gym_env.utils import robot_utils
from gym_env.utils import object_utils
from gym_env.utils import placement_utils
from gym_env.utils import image_utils
from gym_env.utils import helpers


class PlacementEnv4(base_env.BaseGymEnv):
    """
    Franka gym environment for one arm, positional action space, and observation space consisting of the rgb
    camera render data. Set up to perform the placement of a peg that starts in the robots hand into the slot
    correct slot out of several options.
    """
    def __init__(self, config_file):
        """
        Initializes a franka gym environment for one arm. Sets up parameters required to load the scene - 
        reset() must be called to load the environment.

        Parameters
        ----------
        config_file : str
            Path to yaml file consisting of simulation configuration.
        """
        super().__init__(config_file)
        self.plane_id, self.slots, self.pegs, self.table_id = None, None, None, None
        self.table_height, self.peg_heights = 0, None
        self.franka_robot = None
        self.placements_before_termination, self.successful_placements = 0, 0
        self.holding_peg = False
        self.dropped_peg_timer = 0
        self.env_step = 0
        self.inserted = None
        # assert self.render_mode == 'rgb_array', \
        #     'This environment\'s observation space consists of both the rgb and depth data so the render mode must ' \
        #     'be \'rgb_and_depth_arrays\'.'
        self.peg_dimensions = self.load_peg_dimensions()

    def load_peg_dimensions(self): 
        """
        Loads the dimensions of the pegs for use when loading the pegs.
        """
        dimensions_file = os.environ['path-to-pegs'] + "dimensions.yaml"
        dimensions = {}
        with open(dimensions_file) as config:
            try:
                dimensions = base_env.yaml.safe_load(config)
            except Exception as err:
                print('Error Configuration File:{}'.format(err))
                raise err
        return dimensions

    def reset_camera(self):
        """
        Resets the camera view and projection matrices for all cameras according to the values specified in the config file.
        """
        self.camera = tuple( base_env.Camera(camera_config, self.bullet_client) for camera_config in self.config_file['cameras'] )
    
    def render(self, mode=None):
        """
        Renders the current state of the environment.

        Parameters
        ----------
        mode : str
            The desired render mode (i.e. human or rgb).
        Returns
        -------
        tuple
            The elements of the tuple depend on the value of the mode parameter. If mode='rgb_and_depth_arrays' they are tuples 
            of numpy arrays where the first item is the rgb render data (shape is height_resolution x width_resolution x 3) 
            and thesecond item is the depth render data (shape is height_resolution x width_resolution). If mode='rgb_array' 
            they are numpy arrays of the rgb render data (shape is height_resolution x width_resolution x 3). 
            If mode='human' they are None.
        """
        if mode is None:
            mode = self.render_mode
        return tuple( camera.render(self.sim_mode, mode) for camera in self.camera )

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
            whether any arm has collided with a bin, whether the insertion limit has been exceeded, the number of
            completed insertions, and if the peg has been dropped on the ground.
        """
        info = {'slot-collision': False, 'placement-limit-exceeded': False, 
            'successful-placement': False, 'dropped-peg': False, 'table-collision': False,
            'failed-placement': False, 'positioned-peg': False}        
        if self.config_file['termination']['terminate-on-collision-with-slot']['enable']:
            info['slot-collision'] = robot_utils.check_collision_between_robot_and_object(
                self.bullet_client,
                self.franka_robot.robot_id,
                self.slot,
            )
        if self.config_file['termination']['terminate-on-collision-with-table']['enable']:
            info['table-collision'] = robot_utils.check_collision_between_robot_and_object(
                self.bullet_client,
                self.franka_robot.robot_id,
                self.table_id,
            )
        if self.config_file['termination']['terminate-after-successful-placements']['enable']:
            for i in range(len(self.pegs)):
                if self.inserted[i]:
                    continue
                placement_info = placement_utils.check_peg_placement(
                    self.bullet_client,
                    self.pegs[i],
                    self.slot,
                    self.peg_dimensions,
                    self.peg_geometry[i],
                    self.peg_scale_factor[i],
                    self.angle_tol,
                    self.dist_tol,
                    i
                )
                if placement_info['successful-placement']:
                    self.successful_placements += 1
                for key, val in placement_info.items():
                    info[key] |= val
            info['placement-limit-exceeded'] = self.successful_placements >= self.placements_before_termination
        return info['slot-collision'] or info['placement-limit-exceeded'] or info['dropped-peg'] or info['table-collision'], info        

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
        # if info['slot-collision']:
        #     return -1
        # if info['successful-placement']:
        #     return 1
        return -0.1

    def get_stage_reward(self, info, epsilon1=0.004, epsilon2=-0.01, lambda_=2, c_a=0.5, c_b=0.25):
        d_xy, d_xyz, d_z, d_q = self.calc_peg_slot_ref_pos()
        #  print('dxy {}, dxyz {}, dz {}, dq {}'.format(d_xy, d_xyz, d_z, d_q))

        if d_xy > epsilon1:  # r_reach
            r_reach = 1 - 0.5*(math.tanh(lambda_*d_xyz/self.init_d_xyz) + math.tanh(lambda_*d_xy/self.init_d_xy))
            if d_xy > 0.1 or d_z>0.2:
                info['too_far'] = True
                return r_reach, True, info  # too far away
            else:
                return r_reach, False, info
        else:
            if d_z > 0:  # r_alignment
                r_alignment = 2 - c_a * d_xy/epsilon1 - c_b * d_q
                if d_z > 0.2:  # too far away
                    info['too_far'] = True
                    return r_alignment, True, info
                else:
                    return r_alignment, False, info
            elif d_z > epsilon2:  # r_insertion
                return 2 + 2*d_z/epsilon2, False, info
            else:  # r_finish
                return 10, True, info

    def get_state(self):
        """
        Gets the rgbcamera render data as a representation of the current state of the env.

        Returns
        -------
        tuple
            A tuple of numpy arrays that correspond to the rgbcamera data for each configured camera. 
            RGB array has shape (height x width x 3) where the 3 corresponds to the number of colors. 
            In general terms it is an observation of the current state of the franka arm in the environment.
            (same format as described by this envs observation_space).
        """
        for i in range(len(self.config_file['cameras'])):
            if self.config_file['cameras'][i]['eye-hand-configuration'] == 'eye_in_hand':
                robot_utils.mount_camera_on_robot(self.franka_robot, self.cameras[i])
        cam_images = self.render(self.render_mode)
        joint_states = self.franka_robot.get_joint_states()
        ee_states = self.franka_robot.get_end_effector_states()
        if self.render_mode == 'rgb_array':
            cam_images = image_utils.numpy_img_normalize(cam_images[0]).astype(np.float32)
        # robot_pos = np.concatenate((
        #         joint_states['positions'], joint_states['velocities'], [j/100 for j in joint_states['torques']], ee_states['position'],
        #         ee_states['orientation'], ee_states['linear-velocity'], ee_states['angular-velocity']
        #     )).astype(np.float32)
        robot_pos = np.concatenate((
            joint_states['positions'], joint_states['velocities'], [j / 100 for j in joint_states['torques']]
        )).astype(np.float32)
        # observation = [img, robot_pos]
        observation = {
            'robot_pos': robot_pos,  # np.expand_dims(robot_pos, axis=0).astype(np.float32),
            'observation': cam_images,  # np.expand_dims(img, axis=0)
        }
        # print(observation['observation'].shape)
        return observation

    def initialize_scene(self):
        """
        Initializes all the aspects of the scene. Loads the plane, slot, peg, table
        and franka robot and sets them all as class attributes. The finger indices and grasptarget index passed to the
        picking robot constructor are as defined in the panda.urdf file (urdf representation of the robot). Also 
        initializes the space attributes as specified in set_space_attributes().
        """
        path_to_plane_urdf = os.environ['path-to-assets'] + 'plane/plane.urdf'
        self.successful_placements = 0
        self.plane_id = scene_utils.load_plane(self.bullet_client, path_to_plane_urdf)
        self.table_id, self.table_height = scene_utils.load_table(self.bullet_client)
        # self.bin_with_objs_id, _ = bin_utils.load_bin(
        #     self.bullet_client,
        #     self.config_file['scene']['bin'],
        #     False
        # )
        self.franka_robot = robot_utils.load_robot(
            bullet_client=self.bullet_client,
            robot_config=self.config_file['robot'],
            finger_indices=[9, 10],
            grasptarget_index=11
        )
        robot_utils.reset_arm(self.franka_robot,self.config_file['robot'])
        self.slot,info = placement_utils.load_base_slot(
            self.bullet_client, 
            self.config_file['scene']['peg-slot']
        )
        self.peg = placement_utils.load_peg_prev(
            self.bullet_client,
            self.config_file['scene']['peg-slot'],
            self.franka_robot
        )
        # peg_positions = helpers.get_obj_spawn_positions(self.config_file['scene']['bin']['position'], len(info))
        # self.pegs, self.peg_geometry, self.peg_scale_factor = [], [], []
        # for i,config in enumerate(info):
        #     config['peg'] = {
        #         'position': peg_positions[i],
        #         'orientation': [45,45,0],
        #         'spawn-in-hand': {'enabled':False}
        #     }
        #     try:
        #         config['mass'] = self.config_file['scene']['peg-slot']['mass'][i]
        #     except:
        #         pass
        #     peg = placement_utils.load_peg(
        #         self.bullet_client,
        #         config,
        #         self.franka_robot,
        #         self.peg_dimensions
        #     )
        #     self.pegs.append(peg)
        #     self.peg_geometry.append(config['geometry'])
        #     self.peg_scale_factor.append(config['scale-factor'])
        # self.inserted = [False]*len(self.pegs)
        # try:
        #     self.angle_tol = self.config_file['termination']['terminate-after-successful-placements']['failed-placement-angle-tolerance']
        # except:
        #     self.angle_tol = 360
        # try:
        #     self.dist_tol = self.config_file['termination']['terminate-after-successful-placements']['failed-placement-distance-tolerance']
        # except:
        #     self.dist_tol = 1
        # object_utils.allow_objects_to_fall(self.bullet_client, self.config_file['scene']['steps-for-objects-to-fall'])
        

    def calc_peg_slot_ref_pos(self, peg_bottom=0.07, slot_top=0.09, calll=''):
        peg_pos, peg_orn = self.bullet_client.getBasePositionAndOrientation(self.peg)
        peg_bottom_point = placement_utils.calc_rigid_body_coors(self.bullet_client, self.peg, peg_bottom)
        slot_pos, slot_orn = self.bullet_client.getBasePositionAndOrientation(self.slot)
        # slot_top_point = placement_utils.calc_rigid_body_coors(self.bullet_client, self.slot, slot_top)

        peg_pitch, peg_roll, peg_yaw = self.bullet_client.getEulerFromQuaternion(peg_orn)  # pitch --> pi, roll --> 0
        slot_pitch, slot_roll, slot_yaw = self.bullet_client.getEulerFromQuaternion(slot_orn)
        # print('peg pitch / slot pitch {} / {}, peg roll / slot roll {} / {}'.format(peg_pitch-math.pi, slot_pitch, peg_roll, slot_roll))
        cylinder_slot_in_base = [0.017, -0.017, 0]
        slot_top_point = np.array(slot_pos) + placement_utils.rotate_vector(cylinder_slot_in_base, slot_orn) + np.array([0,0,slot_top])
        d_q = (math.tanh(peg_pitch-math.pi)**2 + math.tanh(peg_roll)**2)  # (math.cos(peg_yaw - slot_yaw) + 1)/2
        px, py, pz = peg_bottom_point
        sx, sy, sz = slot_top_point
        d_xy = math.sqrt((sx - px) ** 2 + (sy - py) ** 2)
        d_z = pz - sz
        # print('call {}, pz {}, sz {}, dz {}'.format(calll, pz, sz, d_z))
        d_xyz = math.sqrt(d_xy ** 2 + d_z ** 2)
        # self.bullet_client.addUserDebugLine(slot_top_point, slot_top_point + np.array([0,0,0.01]), lineColorRGB=[1, 0, 0], lineWidth=3.0)
        # print('dxy {:+.3f}, dxyz {:+.3f}, dz {:+.3f}, dq {:+.3f}'.format(d_xy, d_xyz, d_z, d_q))
        return d_xy, d_xyz, d_z, d_q

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
        pbt = self.config_file['termination']['terminate-after-successful-placements']['placements-before-termination']
        self.placements_before_termination = pbt #len(self.pegs) if len(self.pegs) < ibt else ibt
        if self.sim_mode == 'gpu-gui':
            scene_utils.enable_rendering_in_gui(self.bullet_client, self.config_file['simulation']['enable-windows'])
        self.set_space_attributes()
        self.init_d_xy, self.init_d_xyz, _, _ = self.calc_peg_slot_ref_pos(calll='reset')
        return self.get_state()

    def set_space_attributes(self):
        """
        Defines the action space, observation space, and reward range. The action space supports positional control for
        a franka arm (9 dimension box with respective lower and upper positional bounds for each joint or end
        effector). Observation space consists of the rgb camera render data for each configured camera.
        """
        self.action_space = spaces.Box(
            low=np.array(self.franka_robot.lower_limits),
            high=np.array(self.franka_robot.upper_limits),
            dtype=np.float32
        )
        self.observation_space = spaces.Tuple(tuple(
            spaces.Box(
                low=0,
                high=255,
                shape=(
                    camera_config['height-resolution'],
                    camera_config['width-resolution'],
                    3
                ),
                dtype=np.uint8
            ) for camera_config in self.config_file['cameras']
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
        # self.franka_robot.apply_action_pos(action)
        # print(action)
        self.franka_robot.apply_action_vel(list(action) + [0, 0, 0])  # add default finger vel.
        # self.franka_robot.apply_action_vel_impedance(action)
        observation = self.get_state()
        info = {'finished': False, 'too_far': False}
        # done, info = self.check_termination()
        reward, done, info = self.get_stage_reward(info)
        self.env_step += 1
        return observation, reward, done, info