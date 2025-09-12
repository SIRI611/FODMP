import os
import numpy as np
import math
from gym import spaces
import gym_env
from gym_env.envs import base_env
from gym_env.utils import bin_utils
from gym_env.utils import scene_utils
from gym_env.utils import robot_utils
from gym_env.utils import object_utils
from gym_env.utils import placement_utils
from gym_env.utils import metaworld_obj_utils
from gym_env.utils import image_utils
from gym_env.utils import helpers


class BallIntoGoalEnv(base_env.BaseGymEnv):
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
        self.max_eps_length = 150

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
        #     self.config_file['scene']['bin],
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
        self.franka_robot.open_gripper(0.02)
        self.container_id, self.container_info = metaworld_obj_utils.load_drawer(
            self.bullet_client,
            self.config_file['scene']['drop-containers']
        )

        # self.objects = metaworld_obj_utils.load_objects_to_collect(
        #     self.bullet_client,
        #     self.config_file['scene']['objects']
        # )
        self.target_objects, self.disruptive_objects = metaworld_obj_utils.load_target_and_disruptive_objects(
            self.bullet_client, self.config_file['scene']['target-object'],
            self.config_file['scene']['disruptive-objects'], target_color=[0.43921569, 0.38823529, 0.25098039, 1])

        # metaworld_obj_utils.load_background_objects(
        #     self.bullet_client,
        #     self.config_file['scene']['backgrounds']
        # )
        self.reset_camera()
        self.set_space_attributes()
        if self.sim_mode == 'gpu-gui':
            scene_utils.enable_rendering_in_gui(self.bullet_client, self.config_file['simulation']['enable-windows'])
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
        self.env_step = 0
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
        self.franka_robot.open_gripper(0.02)
        self.current_target_obj_index = 0
        # super().init_scene()
        # self.initialize_scene()
        # reset robot
        # reset object placement
        # for obj in self.target_objects:
        #     metaworld_obj_utils.reset_object(self.bullet_client, obj['obj_id'], obj['positions'])
        for i in range(len(self.target_objects)):
            self.target_objects[i]['status'] = 0
        metaworld_obj_utils.reset_target_disruptive_objects(self.bullet_client, self.target_objects, self.disruptive_objects)

        # # set velocity for ball
        # ball_id = self.target_objects[0]['obj_id']
        # self.bullet_client.resetBaseVelocity(
        #     ball_id,
        #     linearVelocity=[-1.0, 1.0, 0.0],   # m/s in x direction
        #     angularVelocity=[0.0, 0.0, 0.0]
        # )

        for _ in range(100):
            self.bullet_client.stepSimulation()
        # self.init_d_xy, self.init_d_xyz, _, _ = self.calc_peg_slot_ref_pos(calll='reset')
        # metaworld_obj_utils.remove_objects(self.bullet_client, self.target_objects)
        # metaworld_obj_utils.remove_objects(self.bullet_client, self.disruptive_objects)

        # self.target_objects, self.disruptive_objects = metaworld_obj_utils.load_target_and_disruptive_objects(
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

    def check_success(self, thres=0.16):
        # # if cup is on top of the drawer, return success.
        # pos_drawer, _ = self.bullet_client.getBasePositionAndOrientation(self.container_id)
        # pos_drawer = np.array(pos_drawer)
        # pos_drawer[0] = pos_drawer[0] - 0.2
        # pos_hammer, _ = self.bullet_client.getBasePositionAndOrientation(self.target_objects[0]['obj_id'])
        # pos_hammer = np.array(pos_hammer)
        # dist_linear = np.linalg.norm(pos_hammer[0:2] - pos_drawer[0:2])
        # dist_z = pos_hammer[2] - pos_drawer[2]
        # # print('dist linear {}, dist z {}'.format(dist_linear, dist_z))
        # # if dist_linear <= 0.05 and dist_z > 0.02 and dist_z < 0.071:
        # #     return True
        # # else:
        # #     return False
        return self.ball_stop(self.target_objects[0]['obj_id'])
        # return self.ball_inside_goal(self.target_objects[0]['obj_id'])

    def get_reward(self):
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
        info = {"success": self.check_success()}
        done = False
        if info['success'] or self.env_step >= self.max_eps_length:
            done = True
        return -0.1, done, info

    def get_image_obs(self, patch_size_half=20):
        cam_images = self.render(self.render_mode)  # 0:  1 640*480*3, 128 * 128*3 *3

        obj_poses = np.zeros((4,3))
        obj_patch_images=[]
        obj_idx = 0
        for obj in self.target_objects + self.disruptive_objects:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(obj['obj_id'])
            obj_poses[obj_idx] = np.array(pos)
            obj_idx += 1

        return obj_poses, np.array(cam_images[0]), np.array(cam_images[1]), np.array(cam_images[2])
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
        obj_poses, top_down_view_img, corner1_img, corner2_img = self.get_image_obs()

        # from PIL import Image
        # for i in range(len(obj_img_patches)):  # [patch 1, patch 2, ...]
        #     im = Image.fromarray(obj_img_patches[i], mode='RGB')
        #     im.save('test_patch_{}.jpg'.format(i))
        # im2 = Image.fromarray(top_down_view_img, mode='RGB')
        # im2.save('topdownview.jpg')
        joint_states = self.franka_robot.get_joint_states()
        ee_states = self.franka_robot.get_end_effector_states()
        gripper_dist = self.franka_robot.get_gripper_distance()
        obj_poses_flat = [j for j in obj_poses.flatten()]

        # append container pos
        target_pos = self.container_info['container']['position']
        obj_poses_flat = obj_poses_flat + target_pos
        # print(obj_poses_flat)
        continuous_states = list(ee_states['position']) + [gripper_dist] + list(joint_states['positions']) + list(joint_states['velocities'])+ obj_poses_flat
        continuous_states = np.array(continuous_states).astype(np.float32)
        # return 'state': 34dim_vector; 'obj_imgs': [40*40*3, 40*40*3, 40*40*3, 40*40*3], ''top_view_img': 128*128*3
        # print(continuous_states)
        observation = {
            'state': continuous_states,  # np.expand_dims(robot_pos, axis=0).astype(np.float32),
            'top_view_img': top_down_view_img,
            'corner1_img': corner1_img,
            'corner2_img': corner2_img
        }
        # cam_images=[top_down_view_img, corner1_img, corner2_img]
        # from PIL import Image
        # for i in range(len(cam_images)):  # [patch 1, patch 2, ...]
        #     im = Image.fromarray(cam_images[i], mode='RGB')
        #     im.save('cams_{}.jpg'.format(i))
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

    def apply_discrete_skills(self, skill_index, max_sim_steps=50, grasping_z=0.00, place_down_z=0.35):
        # 0 move, do nothing
        # 1 pick up: go to z = 0.8164, close gripper, back to original xyz
        # 2 place down: go to z =1.2, then, open gripper, back to original xyz
        # 3 hand over from robot 2 to robot 1
        current_gripper_pos, _, _ = self.franka_robot.get_gripper_state()
        x, y, z = current_gripper_pos
        if skill_index == 1:  # pre-push
            if not self.franka_robot.gripper_is_open:
                self.franka_robot.open_gripper(0.02)
            self.franka_robot.go_to_target_ee_pos([x, y, grasping_z], sim_steps=400, top_down_orn_degrees=[0, 180, 70])
            # self.franka_robot.close_gripper(closed_position=0.05)
            # check if object picked up.
        # elif skill_index == 2:
        #     self.franka_robot.go_to_target_ee_pos([x, y, place_down_z], sim_steps=50)
        #     self.franka_robot.go_to_target_ee_pos([x, y, place_down_z-0.1], sim_steps=50)  # place_down_z-0.2. put sim_steps = 50 when demo!!!!!!!!!!
        #     self.franka_robot.open_gripper()
            # metaworld_obj_utils.reset_object(self.bullet_client, self.obj_id_to_remove, [100, 100, 0.1])
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

    def stop_object(self, obj_info, grasping_z=0.002, dist_thres=0.005, max_approach_steps=60,
                    v_ee_max=0.5,     # EE max XY speed (m/s): tune to your controller
                    latency=0.05,     # control/obs latency margin (s)
                    lead=0.02,        # meet the ball slightly ahead along its heading (m)
                    g=9.81):
        """
        Intercept the rolling ball and stop it: plan to the earliest reachable point
        along its friction-slowed trajectory.
        Returns: ([dx, dy, dz, discrete_skill_index], done_mark)
        """
        # 1) Read ball state
        pos, _ = self.bullet_client.getBasePositionAndOrientation(obj_info['obj_id'])
        vel, _ = self.bullet_client.getBaseVelocity(obj_info['obj_id'])
        x, y, z = pos
        vx, vy, vz = vel
        v0 = float((vx**2 + vy**2) ** 0.5)

        # 2) Effective friction μ from table dynamics (fallback to a sane default)
        try:
            di = self.bullet_client.getDynamicsInfo(self.table_id, -1)
            lat_mu = float(di[1])
            roll_mu = float(di[6]) if len(di) > 6 and di[6] is not None else 0.0
            spin_mu = float(di[7]) if len(di) > 7 and di[7] is not None else 0.0
            mu_eff = max(1e-4, 0.7*lat_mu + 0.2*roll_mu + 0.1*spin_mu)
        except Exception:
            mu_eff = 0.15  # default; tune to match observed decel

        a = max(1e-6, mu_eff * g)  # deceleration magnitude

        # Early exit: ball basically stopped → go to its current XY
        if v0 < 1e-4:
            target_xy = np.array([x, y], dtype=float)
        else:
            dhat = np.array([vx, vy], dtype=float) / v0          # heading
            t_stop = v0 / a                                       # time to stop
            # parametric position along path under constant decel
            def pos_at_t(t):
                t = float(np.clip(t, 0.0, t_stop))
                s = max(0.0, v0*t - 0.5*a*t*t)                   # arc length traveled
                return np.array([x, y]) + dhat * s

            # 3) choose earliest reachable intercept time t
            # current EE XY
            ee_states = self.franka_robot.get_end_effector_states()
            ex, ey, ez = map(float, ee_states['position'])
            dt_plan = 0.02
            t = 0.0
            best_t = None
            while t <= t_stop + 1e-9:
                px, py = pos_at_t(t)
                dist = float(np.hypot(px - ex, py - ey))
                t_needed = dist / max(1e-6, v_ee_max) + latency
                if t_needed <= t:
                    best_t = t
                    break
                t += dt_plan

            if best_t is None:
                # can’t meet before stop → aim at stop point
                s_stop = v0*v0 / (2.0*a)
                target_xy = np.array([x, y]) + dhat * s_stop
            else:
                # lead slightly along heading to meet the front of the ball
                target_xy = pos_at_t(best_t) + dhat * lead

        # 4) Move toward intercept point at a blocking height
        intercept_z = grasping_z + 0.05
        goal_xyz = np.array([target_xy[0], target_xy[1], intercept_z], dtype=float)

        delta_xyz, _, dist_linear, _ = self.franka_robot.approach_target(goal_xyz)
        dx, dy, dz = delta_xyz

        # Your existing done logic
        discrete_skill_index = 0
        done_mark = False
        self.robot_approach_target_attempt_step += 1
        if self.robot_approach_target_attempt_step >= max_approach_steps or dist_linear <= dist_thres:
            self.robot_approach_target_attempt_step = 0
            done_mark = True

        return [dx, dy, dz, discrete_skill_index], done_mark
  
    def intercept_hit_and_brake(self, obj_info,
                                z_block=0.70,           # EE height for contact
                                lead_forward=0.05,      # go THROUGH the ball so we surely hit (meters)
                                latency=0.03,           # control/obs latency buffer
                                v_ee_max=0.9,           # EE planar speed limit (m/s)
                                a_ee_max=3.0,           # EE planar accel limit (m/s^2)
                                dt_cmd=1/60.0,          # one control tick (s)
                                mu_eff=None, g=9.81,    # ball decel model: a = mu_eff*g
                                dist_thres=0.01,        # done when closer than this (m)
                                max_approach_steps=600,
                                brake_time=0.4,         # seconds to ramp EE speed to 0 after contact
                                contact_dist=0.03       # consider “contact” if closer than this (m)
                                ):
        """
        Plan earliest intercept, intentionally pass through the ball (no avoidance),
        detect contact, then gradually slow the EE (braking profile) while keeping contact.
        Returns: ([dx, dy, dz, discrete_skill_index], done_mark)
        """
        # ---------- Read ball planar state ----------
        ball_id = obj_info['obj_id']
        (bx, by, bz), _ = self.bullet_client.getBasePositionAndOrientation(ball_id)
        (bvx, bvy, bvz), _ = self.bullet_client.getBaseVelocity(ball_id)
        v0 = float(math.hypot(bvx, bvy))
        p0 = np.array([bx, by], dtype=float)

        # ---------- Effective friction (fallback if not set) ----------
        if mu_eff is None:
            try:
                di = self.bullet_client.getDynamicsInfo(self.table_id, -1)
                lat_mu = float(di[1])
                roll_mu = float(di[6]) if len(di) > 6 and di[6] is not None else 0.0
                spin_mu = float(di[7]) if len(di) > 7 and di[7] is not None else 0.0
                mu_eff = max(1e-4, 0.7*lat_mu + 0.2*roll_mu + 0.1*spin_mu)
            except Exception:
                mu_eff = 0.15
        a_ball = max(1e-6, mu_eff * g)

        if v0 < 1e-4:
            dhat = np.zeros(2)
            t_stop = 0.0
            def pos_at_t(t): return p0.copy()
        else:
            dhat = np.array([bvx, bvy]) / v0
            t_stop = v0 / a_ball
            def pos_at_t(t):
                t = float(np.clip(t, 0.0, t_stop))
                s = max(0.0, v0*t - 0.5*a_ball*t*t)
                return p0 + dhat * s

        # ---------- EE current state ----------
        ee = self.franka_robot.get_end_effector_states()
        ex, ey, ez = map(float, ee['position'])

        # ---------- Earliest reachable intercept (no avoidance) ----------
        if t_stop <= 0.0:
            target_xy = p0
        else:
            t = 0.0
            dt_plan = 0.02
            best_t = None
            while t <= t_stop + 1e-9:
                px, py = pos_at_t(t)
                dist = float(math.hypot(px - ex, py - ey))
                t_needed = dist / max(1e-6, v_ee_max) + latency
                if t_needed <= t:
                    best_t = t
                    break
                t += dt_plan

            if best_t is None:
                # Go to stop point if we can't meet earlier
                s_stop = v0*v0 / (2.0*a_ball) if v0 >= 1e-4 else 0.0
                target_xy = p0 + dhat * s_stop
            else:
                # IMPORTANT: push THROUGH the ball, not before it
                target_xy = pos_at_t(best_t) + dhat * lead_forward

        goal_xyz = np.array([target_xy[0], target_xy[1], z_block], dtype=float)

        # ---------- Distance & direction to target ----------
        p_now = np.array(self.franka_robot.get_end_effector_states()['position'], dtype=float)
        dvec = goal_xyz - p_now
        dist = float(np.linalg.norm(dvec))
        if dist < 1e-9:
            dirv = np.zeros(3)
        else:
            dirv = dvec / dist

        # ---------- Contact detection (end-effector vs ball) ----------
        # Your robot is loaded with grasptarget_index=11 (panda hand link) :contentReference[oaicite:1]{index=1}
        ee_link = 11
        contacts = self.bullet_client.getContactPoints(self.franka_robot.robot_id, ball_id, linkIndexA=ee_link)
        # Also consider "near contact" by distance for robustness:
        near_contact = dist <= contact_dist

        # State bookkeeping
        if not hasattr(self, "hit_contacted"):  # safety in case not initialized
            self.hit_contacted = False
            self.hit_timer = 0

        if contacts or near_contact:
            self.hit_contacted = True

        # ---------- Velocity command with follow-through then braking ----------
        if not self.hit_contacted:
            # Pre-contact: fast but bounded, with decel as you approach (trapezoid profile)
            # v_cmd = min(v_max, sqrt(2 a_max * dist))
            v_cmd = min(v_ee_max, math.sqrt(max(0.0, 2.0 * a_ee_max * dist)))
        else:
            # Post-contact: follow-through briefly, then ramp down over brake_time
            self.hit_timer += dt_cmd
            tau = min(1.0, self.hit_timer / max(1e-4, brake_time))
            # cosine ramp down: starts near v_ee_max*0.8, ends at 0
            v_cmd = 0.8*v_ee_max * 0.5*(1.0 + math.cos(math.pi * tau))
            # If the ball is basically stopped, clamp more strongly
            speed_ball = float(math.hypot(bvx, bvy))
            if speed_ball < 0.02:
                v_cmd *= 0.25

        step = v_cmd * dt_cmd
        if step > dist:
            step = dist
        delta = dirv * step
        dx, dy, dz = map(float, delta)
        discrete_skill_index = 0  # still position-control move

        # ---------- Done criteria ----------
        self.robot_approach_target_attempt_step += 1
        done_mark = False
        # “Done” when we’ve arrived OR we finished braking after contact
        arrived = dist <= dist_thres
        finished_braking = self.hit_contacted and (self.hit_timer >= brake_time)

        if (self.robot_approach_target_attempt_step >= max_approach_steps) or arrived or finished_braking:
            self.robot_approach_target_attempt_step = 0
            # reset hit state for next call/episode
            self.hit_contacted = False
            self.hit_timer = 0
            done_mark = True

        return [dx, dy, dz, discrete_skill_index], done_mark
    def pick_up_object(self, obj_info, grasping_z=0.001, dist_thres=0.005, max_approach_steps=60):
        # return action: dx, dy, dz, discrete_skill_index
        # approach object until x, y meets, and z=grasping_z+0.1
        """
        Return action: dx, dy, dz, discrete_skill_index
        approach object until x, y meets, and z=grasping_z+0.1
        """
        
        pos, _ = self.bullet_client.getBasePositionAndOrientation(obj_info['obj_id'])
        x, y, z = pos
        goal_pos, _ = self.bullet_client.getBasePositionAndOrientation(self.container_id)
        pre_push_x, pre_push_y = metaworld_obj_utils.AB_line_dist(goal_pos, pos, -0.16, fromA_B=1)
        delta_xyz, _, dist_linear, _ = self.franka_robot.approach_target(np.array([pre_push_x, pre_push_y, 0.15]))
        # print('reach grasping object, x,y,z [{}, {}, {}], dist_linear {}'.format(x, y, z, dist_linear))
        discrete_skill_index = 0
        done_mark = False
        self.robot_approach_target_attempt_step += 1
        if self.robot_approach_target_attempt_step>=max_approach_steps or dist_linear <= dist_thres:
            self.robot_approach_target_attempt_step = 0
            done_mark = True
        dx, dy, dz = delta_xyz

        return [dx, dy, dz, discrete_skill_index], done_mark

    def place_down_to_container(self, place_down_z=0.000, dist_thres=0.1, max_approach_steps=40):
        # return action: dx, dy, dz, discrete_skill_index
        pos, _ = self.bullet_client.getBasePositionAndOrientation(self.container_id)
        x, y, z = pos
        delta_xyz, _, dist_linear, _ = self.franka_robot.approach_target(np.array([x, y, place_down_z]))
        # print('reach container, x,y,z [{}, {}, {}], dist_linear {}'.format(x, y, z, dist_linear))
        discrete_skill_index = 0
        done_mark = False
        self.robot_approach_target_attempt_step += 1
        if self.robot_approach_target_attempt_step>=max_approach_steps:
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
        # if obj_status == 0:
        action, done = self.stop_object(obj_info=self.target_objects[i])
        if done:  # approached the target, can go to status 1
            self.target_objects[i]['status'] = 1
        return action
        # if obj_status == 1:  # need to approach target
        #     action, done = self.pick_up_object(self.target_objects[i])
        #     if done:  # approached the target, can go to status 1
        #         self.target_objects[i]['status'] = 2
        #     return action
        # # if obj_status == 2: # successfully approached the target grasp
        # #     discrete_skill_index = 1  # pre_push
            
        # #     self.target_objects[i]['status'] = 3
        # #     return [action[0], action[1], 0.05, discrete_skill_index]
        # if obj_status == 2: # check if grasp success or not, if success, set status to 2, do placedown, else set status to 0, pickup again
        #     action, done = self.place_down_to_container()  # approach the placedown container only
        #     if done:  # approached the target, can go to status 3
        #         self.target_objects[i]['status'] = 3
        #     return action
        # if obj_status == 3: # successfully approached the container
        #     if self.ball_inside_goal(self.target_objects[i]['obj_id']):  # check if contacting the object during the approaching, if not, start again
        #         discrete_skill_index = 0  # do nothing
        #         self.target_objects[i]['status'] = 10
        #         self.obj_id_to_remove = self.target_objects[i]['obj_id']
        #         # self.current_target_obj_index += 1
        #         return [0, 0, 0, discrete_skill_index]
        #     else:
        #         self.target_objects[i]['status'] = 1  # regrasp again
        #         action, done = self.pick_up_object(self.target_objects[i])
        #         return action
        # return [0,0,0,0]


    def ball_inside_goal(self, target_obj_id, thres=0.16):
        pos_obj, _ = self.bullet_client.getBasePositionAndOrientation(target_obj_id)
        pos_goal, _ = self.bullet_client.getBasePositionAndOrientation(self.container_id)
        # # x0, y0, _ = pos_goal
        # # x, y, _ = pos_obj
        # # x1, y1 = x-x0, y-y0
        # # v = (x1+y1)*0.707107
        # # u = x1/0.707107 - v
        # # print('obj u {}, v{} / goal x +- {}, y +- {}'.format( x, y, 0.105, 0.07505))
        # if x>x0-0.2*0.0707106781186 and x<x0+0.2*0.0707106781186 and y>y0-0.2*0.0707106781186:
        dist_linear = np.linalg.norm(np.array(pos_obj) - np.array(pos_goal))
        # print('dist linear {}'.format(dist_linear))
        if dist_linear<thres:
            return  True
        else:
            return  False

    def ball_stop(self, target_obj_id):
        vel, _ = self.bullet_client.getBaseVelocity(target_obj_id)
        v = np.linalg.norm(np.array(vel))
        return v<0.05

    def step(self, action=None, continuous=False):
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
        self.franka_robot.apply_action_delta(action[0:3], [0,0,0], top_down_orn=True, top_down_orn_degrees=[0, 180, 70])
        if continuous:
            self.apply_discrete_skills(metaworld_obj_utils.continous_action_to_discrete_index(action[3]))
        else:
            self.apply_discrete_skills(action[3])

        # print(done_mark)
        # self.franka_robot.apply_action_vel_impedance(action)
        # delta_action = [0.01, 0.01, 0, 0.1, 1]
        # robot_utils.apply_delta_action(self.bullet_client, self.franka_robot, delta_action)

        observation = self.get_state()
        reward, done, info = self.get_reward()
        if info['success']:
            print("successssssss {}".format(info['success']))
            # let ball into the goal, for visual effect only.
            for _ in range(200):
                self.bullet_client.stepSimulation()

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