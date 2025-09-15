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
from gym_env.utils import collect_and_clear_utils
from gym_env.utils import image_utils
from gym_env.utils import helpers


class PutDonutEnv(base_env.BaseGymEnv):
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
        self.objects = []
        self.finished_objects = []
        super().init_scene()
        self.initialize_scene()
        self.first_init=True

    def initialize_scene(self):
        """
        Initializes all the aspects of the scene. Loads the plane, slot, peg, table
        and franka robot and sets them all as class attributes. The finger indices and grasptarget index passed to the
        picking robot constructor are as defined in the panda.urdf file (urdf representation of the robot). Also
        initializes the space attributes as specified in set_space_attributes().
        """
        self.current_target_obj_index = 0
        path_to_plane_urdf = os.environ['path-to-assets'] + 'plane/planeMesh.urdf'
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
        # print(self.config_file['scene'])
        robot_utils.reset_arm(self.franka_robot, self.config_file['robot'])
        self.franka_robot.open_gripper()
        self.container_id, self.container_info = collect_and_clear_utils.load_plate(
            self.bullet_client,
            self.config_file['scene']['drop-containers']
        )

        # self.objects = collect_and_clear_utils.load_objects_to_collect(
        #     self.bullet_client,
        #     self.config_file['scene']['objects']
        # )
        self.target_objects, self.disruptive_objects = collect_and_clear_utils.load_target_and_disruptive_objects(
            self.bullet_client, self.config_file['scene']['target-object'],
            self.config_file['scene']['disruptive-objects'], target_color=[0.43921569, 0.38823529, 0.25098039, 1])

        collect_and_clear_utils.load_background_objects(
            self.bullet_client,
            self.config_file['scene']['backgrounds']
        )
        self.reset_camera()
        self.set_space_attributes()
        # if self.sim_mode == 'gpu-gui':
        #     scene_utils.enable_rendering_in_gui(self.bullet_client, self.config_file['simulation']['enable-windows'])
        self.first_init=True
        self.robot_approach_target_attempt_step=0

    def reset(self):
        """
        Resets the simulation and class-relevant environment attributes to their default settings. Then reloads the
        scene. Possible to reload the scene with the same type of objects that were in the bin prior to calling reset.

        Returns
        -------
        object
            Observation of the current state of this env in the format described by this envs observation_space.
        """
        if self.first_init:
            self.first_init = False
            return self.get_state()

        self.bullet_client.removeBody(self.franka_robot.robot_id)
        self.franka_robot = robot_utils.load_robot(
            bullet_client=self.bullet_client,
            robot_config=self.config_file['robot'],
            finger_indices=[9, 10],
            grasptarget_index=11
        )
        # print(self.config_file['scene'])
        robot_utils.reset_arm(self.franka_robot, self.config_file['robot'])
        self.franka_robot.open_gripper()
        self.current_target_obj_index = 0
        # super().init_scene()
        # self.initialize_scene()
        # reset robot
        # reset object placement
        # for obj in self.target_objects:
        #     collect_and_clear_utils.reset_object(self.bullet_client, obj['obj_id'], obj['positions'])
        for i in range(len(self.target_objects)):
            self.target_objects[i]['status'] = 0
        collect_and_clear_utils.reset_target_disruptive_objects(self.bullet_client, self.target_objects, self.disruptive_objects)
        for _ in range(100):
            self.bullet_client.stepSimulation()
        # self.init_d_xy, self.init_d_xyz, _, _ = self.calc_peg_slot_ref_pos(calll='reset')
        # collect_and_clear_utils.remove_objects(self.bullet_client, self.target_objects)
        # collect_and_clear_utils.remove_objects(self.bullet_client, self.disruptive_objects)

        # self.target_objects, self.disruptive_objects = collect_and_clear_utils.load_target_and_disruptive_objects(
        #     self.bullet_client, self.config_file['scene']['target-object'],
        #     self.config_file['scene']['disruptive-objects'])

        return self.get_state()


    def reset_camera(self):
        """
        Resets the camera view and projection matrices for all cameras according to the values specified in the config file.
        """
        self.camera = tuple(
            base_env.Camera(camera_config, self.bullet_client) for camera_config in self.config_file['cameras'])

    def render(self, mode=None):
        if mode is None:
            mode = self.render_mode
        return tuple(camera.render(self.sim_mode, mode) for camera in self.camera)

    def check_termination(self):
        if len(self.objects)>0 and len(self.objects) == len(self.finished_objects):
            return True
        else:
            return False

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
        done=False
        finished_obj_count = 0
        for obj in self.target_objects:
            if obj['status'] == 10:
                finished_obj_count += 1
        if finished_obj_count == len(self.target_objects):
            done = True
        return -0.1, done, None

    def get_image_obs(self, patch_size_half=20):
        cam_images = self.render(self.render_mode)  # 0:  1 640*480*3, 128 * 128*3 *3
        # from PIL import Image
        # for i in range(len(cam_images)):  # [patch 1, patch 2, ...]
        #     im = Image.fromarray(cam_images[i], mode='RGB')
        #     im.save('cams_{}.jpg'.format(i))
        obj_bbbox = []
        obj_patch_images=[]
        for obj in self.target_objects + self.disruptive_objects:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(obj['obj_id'])
            imv_xy = image_utils.world_coors_to_image_uv(pos[0], pos[1])
            # print(imv_xy)
            obj_bbbox.append([imv_xy[0]/640, imv_xy[1]/480, patch_size_half*2/640, patch_size_half*2/480])
            x_border_left, x_border_right = image_utils.crop_left_right_to_border(imv_xy[0], patch_size_half, cam_images[0].shape[0])
            y_border_top, y_border_down = image_utils.crop_left_right_to_border(imv_xy[1], patch_size_half, cam_images[0].shape[1])
            # print("{},{},{},{}".format(x_border_left, x_border_right, y_border_top, y_border_down))
            obj_patch_images.append(np.array(cam_images[0])[y_border_top:y_border_down,x_border_left:x_border_right,:])

        return obj_patch_images, obj_bbbox, np.array(cam_images[1])
        # if self.render_mode == 'rgb_array':
        #     cam_images = image_utils.numpy_img_normalize(cam_images[0]).astype(np.float32)  # convert to pytorch img format


    def get_state(self):
        """
        34-dim continuous values: {Robot x, y, z, gripper_dist, joint_pos(7), joint_vel (7), obj1~4 image bbox}, 5 images: {obj1~4 image patchs, top-view scene image}

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
        obj_img_patches, obj_bbbox, top_down_view_img = self.get_image_obs()
        # from PIL import Image
        # for i in range(len(obj_img_patches)):  # [patch 1, patch 2, ...]
        #     im = Image.fromarray(obj_img_patches[i], mode='RGB')
        #     im.save('test_patch_{}.jpg'.format(i))
        # im2 = Image.fromarray(top_down_view_img, mode='RGB')
        # im2.save('topdownview.jpg')
        joint_states = self.franka_robot.get_joint_states()
        ee_states = self.franka_robot.get_end_effector_states()
        gripper_dist = self.franka_robot.get_gripper_distance()
        obj_bbbox_flat = [j for obj in obj_bbbox for j in obj]
        # print(obj_bbbox_flat)
        continuous_states = list(ee_states['position']) + [gripper_dist] + list(joint_states['positions']) + list(joint_states['velocities'])+ obj_bbbox_flat
        continuous_states = np.array(continuous_states).astype(np.float32)
        # return 'state': 34dim_vector; 'obj_imgs': [40*40*3, 40*40*3, 40*40*3, 40*40*3], ''top_view_img': 128*128*3
        observation = {
            'state': continuous_states,  # np.expand_dims(robot_pos, axis=0).astype(np.float32),
            'obj_imgs': obj_img_patches,  # np.expand_dims(img, axis=0)
            'top_view_img': top_down_view_img
        }
        # print(observation['observation'].shape)
        # obj_id = self.objects[0]['obj_ids'][0]
        # obj_pos = np.array(self.bullet_client.getBasePositionAndOrientation(obj_id)[0])
        # print(obj_pos)
        # for obj in self.target_objects:
        #     pos, _ = self.bullet_client.getBasePositionAndOrientation(obj['obj_id'])
        #     print('coffee cup')
        #     print(pos)
        # for obj in self.disruptive_objects:
        #     pos, _ = self.bullet_client.getBasePositionAndOrientation(obj['obj_id'])
        #     print('donut')
        #     print(pos)
        return observation

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

    def apply_discrete_skills(self, skill_index, max_sim_steps=50, grasping_z=0.5804-0.625, place_down_z=0.73-0.625):
        # 0 move, do nothing
        # 1 pick up: go to z = 0.8164, close gripper, back to original xyz
        # 2 place down: go to z =1.2, then, open gripper, back to original xyz
        # 3 hand over from robot 2 to robot 1
        current_gripper_pos, _, _ = self.franka_robot.get_gripper_state()
        x, y, z = current_gripper_pos
        if skill_index == 1:
            if not self.franka_robot.gripper_is_open:
                self.franka_robot.open_gripper()
            self.franka_robot.go_to_target_ee_pos([x, y, grasping_z], sim_steps=200)
            self.franka_robot.close_gripper()
            self.franka_robot.go_to_target_z_orn_top_down(z, sim_steps=20)
            # check if object picked up.
        elif skill_index == 2:
            self.franka_robot.go_to_target_ee_pos([x, y, place_down_z], sim_steps=50)
            # self.franka_robot.go_to_target_ee_pos([x, y, place_down_z], sim_steps=50)  # place_down_z-0.2. put sim_steps = 50 when demo!!!!!!!!!!
            collect_and_clear_utils.reset_object(self.bullet_client, self.obj_id_to_remove, [100, 100, 0.1])
            self.franka_robot.open_gripper()
            # self.franka_robot.go_to_target_ee_pos([x, y, z], sim_steps=5)


    # robot 1 object status:
    # 0 default, do pick up object
    # 1 reached grasping target, return action grasp attempted,
    # 2 success grasped, need to check grasp, if success, go to 2, else do pickup obj again
    # 3 sucess reached placedown target, now place down discrete skill
    # 10 success place down

    # robot 2 object status:
    # 0 default
    # 4 reach grasping target
    # 5 success grasped
    # 6 reach robot 1
    # 7 success hand in
    # 8 reach placedown target
    # 10 success place down, finished

    def pick_up_object(self, obj_info, grasping_z=0.5504-0.625, dist_thres=0.005, max_approach_steps=20):
        # return action: dx, dy, dz, discrete_skill_index
        # approach object until x, y meets, and z=grasping_z+0.1
        pos, _ = self.bullet_client.getBasePositionAndOrientation(obj_info['obj_id'])
        x, y, z = pos
        delta_xyz, _, dist_linear, _ = self.franka_robot.approach_target(np.array([x, y, grasping_z+0.2]))
        # print('reach grasping object, x,y,z [{}, {}, {}], dist_linear {}'.format(x, y, z, dist_linear))
        discrete_skill_index = 0
        done_mark = False
        self.robot_approach_target_attempt_step += 1
        if self.robot_approach_target_attempt_step>=max_approach_steps or dist_linear <= dist_thres:
            self.robot_approach_target_attempt_step = 0
            done_mark = True
        dx, dy, dz = delta_xyz

        return [dx, dy, dz, discrete_skill_index], done_mark

    def place_down_to_container(self, place_down_z=0.80-0.625, dist_thres=0.04, max_approach_steps=20):
        # return action: dx, dy, dz, discrete_skill_index
        pos, _ = self.bullet_client.getBasePositionAndOrientation(self.container_id)
        x, y, z = pos
        delta_xyz, _, dist_linear, _ = self.franka_robot.approach_target(np.array([x, y, place_down_z]))
        # print('reach container, x,y,z [{}, {}, {}], dist_linear {}'.format(x, y, z, dist_linear))
        discrete_skill_index = 0
        done_mark = False
        self.robot_approach_target_attempt_step += 1
        if self.robot_approach_target_attempt_step>=max_approach_steps or dist_linear <= dist_thres:
            self.robot_approach_target_attempt_step = 0
            done_mark = True
        dx, dy, dz = delta_xyz
        return [dx, dy, dz, discrete_skill_index], done_mark

    def check_object_on_the_fly(self, obj_id, height_thres=0.70-0.625):
        pos, _ = self.bullet_client.getBasePositionAndOrientation(obj_id)
        return pos[2]>=height_thres

    def expert_action_one_robot(self):
        # return action: dx, dy, dz, discrete_skill_index
        i = self.current_target_obj_index
        # finished_obj_count = 0
        # for obj in self.target_objects:
        #     if obj['status'] == 10:
        #         finished_obj_count += 1
        # if finished_obj_count == len(self.target_objects):
        #     return [0,0,0,0], True
        obj_status = self.target_objects[i]['status']
        if obj_status == 0:  # need to approach target
            action, done = self.pick_up_object(self.target_objects[i])
            if done:  # approached the target, can go to status 1
                self.target_objects[i]['status'] = 1
            return action
        if obj_status == 1: # successfully approached the target grasp
            discrete_skill_index = 1  # grasp skill
            self.target_objects[i]['status'] = 2
            return [0, 0, 0, discrete_skill_index]
        if obj_status == 2: # check if grasp success or not, if success, set status to 2, do placedown, else set status to 0, pickup again
            if self.franka_robot.is_grasping_object(self.target_objects[i]['obj_id']): # object success grasped
                action, done = self.place_down_to_container()  # approach the placedown container only
                if done:  # approached the target, can go to status 3
                    self.target_objects[i]['status'] = 3
                return action
            else:
                self.target_objects[i]['status'] = 0  # regrasp again
                action, done = self.pick_up_object(self.target_objects[i])
                return action
        if obj_status == 3: # successfully approached the container
            discrete_skill_index = 2  # grasp skill
            self.target_objects[i]['status'] = 10
            self.obj_id_to_remove = self.target_objects[i]['obj_id']
            self.current_target_obj_index += 1
            return [0, 0, 0, discrete_skill_index]
        return [0,0,0,0]


    def step(self, action=None):
        """
        Runs one step of the simulation. Applies a position control action to the franka arm and gets an observation.

        Parameters
        ----------
        action : [dx, dy, dz, discrete_skills] 0: move, 1: pick up, 2: place down, 3: insert, 4: strike, 5: zig zag move.
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
        # self.franka_robot.apply_action_vel(list(action) + [0, 0, 0])  # add default finger vel.
        # self.franka_robot.go_to_position([-0.3, 0.1, 1.2])

        # self.franka_robot.go_to_target_ee_pos([x+0.001,y+0.001,z])
        # self.franka_robot.apply_action_delta([0.001, 0.001, 0], [0, 0, 0], top_down_orn=True)
        # if action is None:
        #     action, done_mark = self.expert_action_one_robot()
        # print(action)
        self.franka_robot.apply_action_xyz_orn_top_down(action[0:3])
        self.apply_discrete_skills(action[3])

        # print(done_mark)
        # self.franka_robot.apply_action_vel_impedance(action)
        # delta_action = [0.01, 0.01, 0, 0.1, 1]
        # robot_utils.apply_delta_action(self.bullet_client, self.franka_robot, delta_action)

        observation = self.get_state()
        info = {'finished': False}
        # done, info = self.check_termination()
        # reward, done, info = self.get_stage_reward(info)
        reward, done, info = self.get_reward(info)

        self.env_step += 1
        return observation, reward, done, info


# cameras:
# - scale-depth-data: False
#   eye-hand-configuration: eye_to_hand  # eye_in_hand
#   width-resolution: 128
#   height-resolution: 128
#   view-matrix:
#     eye-position: [-0.5,-0.15,1.55]
#     target-position: [-0.5,-0.14,0.626]
#     up-vector: [0,0,1]
#   projection-matrix:
#     fov: 75
#     near-plane: 0.008 # 0.1 also works
#     far-plane: 2