import os
import math
import numpy as np

class SimpleFrankaRobot:
    """
    Class providing functionality and support for a picking robot. Allows for control and interaction of the robot in a
    simulator (designed for pybullet). Note that a robot being modelled by this class should have an end effector that
    has a gripper consisting of two fingers and a "grasptarget" joint corresponding to the gripper's grasptarget (the
    location between the fingers at which an object should ideally be if it is to be successfully grasped).
    """
    max_steps = 50
    max_grasping_force = 20
    acceptable_joint_proximity = 0.0008
    acceptable_finger_proximity = 0.0000001

    def __init__(
            self,
            bullet_client,
            robot_config,
            finger_indices,
            grasptarget_index
    ):
        """
        Initializes a picking robot with a fixed base at the given position and orientation as stated in the config
        file. First reads the specified initial joint poses as specified in the configuration file and sets the robot
        to that position, then creates a list of the indices of the move-able joints (not including the gripper's
        fingers) in the robot and a list of the max velocities of each move-able joint.
        Note: a negative position index means the joint is fixed.

        Parameters
        ----------
        bullet_client : pybullet_utils.bullet_client.BulletClient
            Pybullet client.
        robot_config : dict
            The robot section of the config file pertaining to the robot that is to be created.
        finger_indices : list
            The index of each finger that makes up the gripper of the picking robot, length 2.
        grasptarget_index : int
            The index of the grasptarget (the location between the fingers of the gripper where an object would ideally
            be if it were to be grasped) - this index is considered the "end effector" in inverse kinematics
            calculations.
        """
        self.bc = bullet_client
        path_to_robot_urdf = os.environ['path-to-robots'] + robot_config['robot-file']
        self.finger_indices = finger_indices
        self.grasptarget_index = grasptarget_index
        self.enable_gripper_force_readings = robot_config['enable-gripper-force-readings']
        self.clip_delta_actions_to_workspace = robot_config['clip-delta-actions-to-workspace']
        base_orn_euler = [math.radians(angle) for angle in robot_config['orientation']]
        base_orn_quaternion = self.bc.getQuaternionFromEuler(base_orn_euler)
        flags = (
                self.bc.URDF_ENABLE_CACHED_GRAPHICS_SHAPES |
                self.bc.URDF_USE_SELF_COLLISION |
                self.bc.URDF_MAINTAIN_LINK_ORDER
        )
        try:
            self.robot_id = self.bc.loadURDF(
                fileName=path_to_robot_urdf,
                basePosition=robot_config['position'],
                baseOrientation=base_orn_quaternion,
                useFixedBase=True,
                flags=flags
            )
            if self.robot_id < 0:
                raise Exception('Cannot load URDF file.')
            else:
                self.moveable_joints, self.max_velocities = [], []
                self.num_joints = self.bc.getNumJoints(self.robot_id)
                for joint in range(self.num_joints):
                    position_index = self.bc.getJointInfo(self.robot_id, joint)[3]
                    if (position_index > -1) and (joint not in self.finger_indices):
                        self.moveable_joints.append(joint)
                for joint in self.moveable_joints + self.finger_indices:
                    joint_max_velocity = self.bc.getJointInfo(self.robot_id, joint)[11]
                    self.max_velocities.append(joint_max_velocity)
                try:
                    self.sigma = robot_config['impedance']['sigma']
                    if not isinstance(self.sigma, list):
                        self.sigma = [self.sigma] * len(self.moveable_joints)
                    self.inner_steps = robot_config['impedance']['inner-steps']
                    self.delay_steps = robot_config['impedance']['delay-steps']
                except:
                    self.sigma = [0] * len(self.moveable_joints)
                    self.inner_steps = 1
                    self.delay_steps = 1
                if self.enable_gripper_force_readings:
                    self.enable_force_sensor_on_fingers()
                self.default_poses = self.read_poses_from_config(robot_config['poses'])
                self.lower_limits, self.upper_limits, self.joint_ranges, self.rest_poses = self.null_space_limits()
                self.reset()
                self.gripper_is_open = True
                self.open_gripper()
        except Exception as ex2:
            self.bc.disconnect()
            print(type(ex2))
            print(ex2.args[0])
            exit()
        desired_orn_euler = [0, 180, 20]  # current_gripper_orn + np.array([0, 0, da])
        self.top_down_orn = [math.radians(angle) for angle in desired_orn_euler]

    def get_joint_states(self):
        """
        get robot state
        :return: 7 joints' pos, velocities.
        """
        joint_states = self.bc.getJointStates(self.robot_id, self.moveable_joints)
        joint_states_dict = {'positions': [joint_states[i][0] for i in range(len(self.moveable_joints))],
                             'velocities': [joint_states[i][1] for i in range(len(self.moveable_joints))],
                             'torques': [joint_states[i][3] for i in range(len(self.moveable_joints))]}
        return joint_states_dict

    def get_end_effector_states(self):
        link_state = self.bc.getLinkState(self.robot_id, linkIndex=11,
                                          computeLinkVelocity=1, computeForwardKinematics=1)
        end_effector_state = {'position': link_state[0],
                              'orientation': link_state[1],
                              'linear-velocity': link_state[6],
                              'angular-velocity': link_state[7]}
        return end_effector_state

    def approach_target(self, target_xyz, target_orn_radians=None, step_size_linear=0.001, step_size_angular=0.017):

        current_gripper_pos, current_gripper_orn, _ = self.get_gripper_state()
        # print('target xyz')
        # print(target_xyz)
        # print('current_gripper_pos')
        # print(current_gripper_pos)
        delta_xyz = (target_xyz - current_gripper_pos)*0.4  # np.sign(target_xyz - current_gripper_pos)*step_size_linear

        dist_linear = np.linalg.norm(target_xyz - current_gripper_pos)
        delta_orn = None
        dist_angular = None
        # if target_orn_radians == None:
        #     self.apply_action_xyz_orn_top_down(delta_xyz)
        # else:
        #     delta_orn = np.sign(target_orn_radians - current_gripper_orn)*step_size_angular
        #     self.apply_action_delta(delta_xyz, delta_orn)
        if target_orn_radians is not None:
            dist_angular = np.linalg.norm(target_orn_radians - current_gripper_orn)
            delta_orn = np.sign(target_orn_radians - current_gripper_orn) * step_size_angular
        return delta_xyz, delta_orn, dist_linear, dist_angular

    def apply_action_xyz_orn_top_down(self, delta_xyz):
        self.apply_action_delta(delta_xyz, [0, 0, 0], top_down_orn=True)

    def apply_action_delta(self, delta_xyz, delta_euler_angles_radians, top_down_orn=False, top_down_orn_degrees=None):
        current_gripper_pos, current_gripper_orn, _ = self.get_gripper_state()
        desired_gripper_pos = current_gripper_pos + np.array(delta_xyz)
        if top_down_orn:
            if top_down_orn_degrees is None:
                self.go_to_target_ee_pos(desired_gripper_pos, self.top_down_orn)
            else:
                # desired_orn_euler = [0, 180, 20]  # current_gripper_orn + np.array([0, 0, da])
                orn = [math.radians(angle) for angle in top_down_orn_degrees]
                self.go_to_target_ee_pos(desired_gripper_pos, orn)
        else:
            desired_orn_euler = current_gripper_orn + np.array(delta_euler_angles_radians)
            self.go_to_target_ee_pos(desired_gripper_pos, desired_orn_euler)

    def go_to_target_z_orn_top_down(self, target_z, sim_steps=20):
        current_gripper_pos, _, _ = self.get_gripper_state()
        current_gripper_pos[2] = target_z
        self.go_to_target_ee_pos(current_gripper_pos, sim_steps=sim_steps)

    def go_to_target_ee_pos(self, target_ee_xyz, target_ee_euler_angles_radians=None, sim_steps=20, top_down_orn_degrees=None):
        # print(target_ee_xyz)
        if target_ee_euler_angles_radians is None:
            if top_down_orn_degrees is None:
                target_ee_euler_angles_radians = self.top_down_orn
            else:
                target_ee_euler_angles_radians = [math.radians(angle) for angle in top_down_orn_degrees]
        desired_gripper_orn = self.bc.getQuaternionFromEuler(target_ee_euler_angles_radians)
        desired_joint_poses = self.get_joint_poses_ik(
            target_ee_position=target_ee_xyz,
            target_ee_orientation=desired_gripper_orn,
        )
        self.apply_action_yk(desired_joint_poses, sim_steps=sim_steps)

    def get_joint_poses_ik(self, target_ee_position, target_ee_orientation=None):
        joint_poses = self.bc.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.grasptarget_index,
            targetPosition=target_ee_position,
            targetOrientation=target_ee_orientation
        )
        return np.array(joint_poses)

    def get_gripper_state(self):
        state = self.bc.getLinkState(self.robot_id, self.grasptarget_index, computeLinkVelocity=True)
        orn = np.array(self.bc.getEulerFromQuaternion(state[1]))
        return np.array(state[0]), orn, np.array(state[6])

    def apply_action_yk(self, target, sim_steps=10):
        self.bc.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.moveable_joints,
            controlMode=self.bc.POSITION_CONTROL,
            targetPositions=target[:len(self.moveable_joints)],
            targetVelocities=[0] * len(self.moveable_joints),
            positionGains=[0.03] * len(self.moveable_joints),
            velocityGains=[1] * len(self.moveable_joints),
            forces=[200.] * len(self.moveable_joints)
        )
        if not self.gripper_is_open:
            # helps keep object in gripper
            max_force = 80
            self.bc.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.finger_indices,
                controlMode=self.bc.POSITION_CONTROL,
                targetPositions=[0, 0],
                forces=[max_force, max_force]
            )
        for _ in range(sim_steps):
            self.bc.stepSimulation()

    def apply_action_to_gripper(self, target_position_a, target_position_b):
        """
        Moves the fingers of the gripper to a target position. Steps the physics engine until either the max number of
        steps are reached or until the position of the fingers are within an acceptable proximity of their target
        position.

        Parameters
        ----------
        target_position_a : float
            Target position of finger a.
        target_position_b : float
            Target position of finger b.
        """
        target_position_a, target_position_b = self.check_target_position_bounds(target_position_a, target_position_b)
        self.bc.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.finger_indices,
            controlMode=self.bc.POSITION_CONTROL,
            targetPositions=[target_position_a, target_position_b],
            forces=[self.max_grasping_force, self.max_grasping_force]
        )
        for i in range(self.max_steps):
            self.bc.stepSimulation()
            target = np.array([target_position_a, target_position_b])
            actual = np.array(self.get_joint_poses()[len(self.moveable_joints):])
            proximity = abs(target - actual)
            if np.array([finger_prox < self.acceptable_finger_proximity for finger_prox in proximity]).all():
                break

    def check_target_position_bounds(self, target_pos_a, target_pos_b):
        """
        Checks if the desired target pose of each finger is valid - if so the original value is returned, otherwise the
        nearest valid pose is returned.

        Parameters
        ----------
        target_pos_a : float
            Target position of end effector a (index 9).
        target_pos_b : float
            Target position of end effector b (index 10).
        Returns
        -------
        tuple
            Two floats, corresponding to the validified target pose of each finger.
        """
        a, b = 0, 1
        fingers_lower_limits = self.lower_limits[len(self.moveable_joints):]
        fingers_upper_limits = self.upper_limits[len(self.moveable_joints):]
        finger_a_ll = fingers_lower_limits[a]
        finger_a_ul = fingers_upper_limits[a]
        finger_b_ll = fingers_lower_limits[b]
        finger_b_ul = fingers_upper_limits[b]
        target_pos_a = min(max(target_pos_a, finger_a_ll), finger_a_ul)
        target_pos_b = min(max(target_pos_b, finger_b_ll), finger_b_ul)
        return target_pos_a, target_pos_b

    def close_gripper(self, closed_position=0):
        """
        Closes the gripper on the robot and updates the gripper's state.
        """
        # closed_position = 0
        self.apply_action_to_gripper(closed_position, closed_position)
        self.gripper_is_open = False

    def enable_force_sensor_on_fingers(self):
        """
        Enables a force/torque sensor on both fingers of the gripper.
        """
        a, b = 0, 1
        self.bc.enableJointForceTorqueSensor(self.robot_id, self.finger_indices[a], enableSensor=1)
        self.bc.enableJointForceTorqueSensor(self.robot_id, self.finger_indices[b], enableSensor=1)

    def get_end_effector_reaction_forces(self):
        """
        Gets the reaction forces sensed by each end effector along the z-axis of their reference frame. Currently the
        other sensed forces [Fx,Fy] and sensed torques [Mx,My,Mz] are ignored.

        Returns
        -------
        tuple
            Two floats corresponding to the reaction forces sensed by each of the gripper's fingers.
        """
        a, b, reaction_forces_index, z = 0, 1, 2, 2
        finger_states = self.bc.getJointStates(self.robot_id, self.finger_indices)
        finger_a_rxn_force_z = finger_states[a][reaction_forces_index][z]
        finger_b_rxn_force_z = finger_states[b][reaction_forces_index][z]
        return finger_a_rxn_force_z, finger_b_rxn_force_z

    def get_gripper_distance(self):
        finger_states = self.bc.getJointStates(self.robot_id, self.finger_indices)
        return finger_states[0][0] + finger_states[1][0]

    def get_joint_poses(self):
        """
        Describes the current state of this robot in terms of the poses of each joint and gripper finger.

        Returns
        -------
        numpy.ndarray
            Array consisting of the pose of each joint and gripper finger in/on the robot. The first
            len(self.moveable_joints) elements correspond to the poses of those joints as ordered in the urdf and the
            last two elements correspond to the pose of the gripper's fingers.
        """
        joints_of_interest = self.moveable_joints + self.finger_indices
        joint_states = self.bc.getJointStates(self.robot_id, joints_of_interest)
        position_observation = [joint_states[joint_num][0] for joint_num in range(len(joints_of_interest))]
        return np.array(position_observation)

    def get_joint_velocities(self):
        """
        Describes the current state of this robot in terms of the velocities of each joint and gripper finger.

        Returns
        -------
        numpy.ndarray
            Array consisting of the velocity of each joint and gripper finger in/on the robot. The first
            len(self.moveable_joints) elements correspond to the velocities of those joints as ordered in the urdf
            and the last two elements correspond to the pose of the gripper's fingers.
        """
        joints_of_interest = self.moveable_joints + self.finger_indices
        joint_states = self.bc.getJointStates(self.robot_id, joints_of_interest)
        velocity_observation = [joint_states[joint][1] for joint in joints_of_interest]
        return np.array(velocity_observation)

    def is_grasping_object(self, obj_id):
        """
        Checks if there are any contact points between a specified object and both of the gripper's fingers.

        Parameters
        ----------
        obj_id : int
            Id of object to check if being grasped.
        Returns
        -------
        bool
            True if both fingers are in contact with the specified object, false otherwise.
        """
        a, b = 0, 1
        contact_points_with_finger_a = self.bc.getContactPoints(
            bodyA=self.robot_id,
            bodyB=obj_id,
            linkIndexA=self.finger_indices[a]
        )
        contact_points_with_finger_b = self.bc.getContactPoints(
            bodyA=self.robot_id,
            bodyB=obj_id,
            linkIndexA=self.finger_indices[b]
        )
        return (len(contact_points_with_finger_a) > 0) and (len(contact_points_with_finger_b) > 0)

    def null_space_limits(self):
        """
        Finds the null space limits (lower limits, upper limits, joint ranges, rest poses) of each moveable joint and
        gripper finger in this robot and sets the instance attributes accordingly. The rest pose is set to the default
        position of the arm.
        Note: a negative position index means the joint is fixed.

        Returns
        -------
        tuple
            Four lists, each consisting of floats. The first element of the tuple is a list that consists of the lower
            limits of each non-fixed joint in the robot, the second is a list that consists of the upper limits of each
            non-fixed joint in the robot, the third is a list that consists of the range of each non-fixed joint in the
            robot, and the fourth is a list that consists of the rest pose (default pose) of each non-fixed joint in
            the robot.
        """
        lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []
        for joint in range(self.num_joints):
            joint_info = self.bc.getJointInfo(self.robot_id, joint)
            position_index = joint_info[3]
            if position_index > -1:
                lower_limits.append(joint_info[8])
                upper_limits.append(joint_info[9])
                joint_ranges.append(joint_info[9] - joint_info[8])
        rest_poses = self.default_poses
        return lower_limits, upper_limits, joint_ranges, rest_poses

    def open_gripper(self, target_position=0.4):
        """
        Opens the gripper. Each finger will move to a certain position. By default each finger moves 0.02 m from its
        origin as this will ensure that the diameter between the fingers is 1 cm larger than the diameter of the
        largest object (3 cm) in both the gym_env.assets.training_objs and the gym_env.assets.testing_objs folders.

        Parameters
        ----------
        target_position : float
            The target position to move each finger to.
        """
        self.apply_action_to_gripper(target_position, target_position)
        # print('gripper open command')
        self.gripper_is_open = True

    def read_poses_from_config(self, poses_from_config):
        """
        Reads the default joint poses specified in the configuration file and loads them into a list in the same order.

        Parameters
        ----------
        poses_from_config : dict
            Default pose of each joint/finger in robot as specified in the config file.
        Returns
        -------
        list
            The default pose of each joint/finger in robot as a list.
        """
        default_poses = []
        joint_num = 0
        while 'joint-' + str(joint_num) in poses_from_config:
            default_poses.append(poses_from_config['joint-' + str(joint_num)])
            joint_num += 1
        finger_num = 0
        while 'finger-' + str(finger_num) in poses_from_config:
            default_poses.append(poses_from_config['finger-' + str(finger_num)])
            finger_num += 1
        return default_poses

    def reset(self):
        """
        Resets all the joints/fingers of this arm to their default position as stored in the instance attribute
        self.default_poses. Resets the gripper's state to open.
        """
        for joint_num in range(len(self.moveable_joints)):
            self.bc.resetJointState(
                bodyUniqueId=self.robot_id,
                jointIndex=self.moveable_joints[joint_num],
                targetValue=self.default_poses[joint_num],
                targetVelocity=0
            )
        self.open_gripper()

    def get_jacobian(self):
        joint_states = self.bc.getJointStates(self.robot_id, range(self.bc.getNumJoints(self.robot_id)))
        joint_infos = [self.bc.getJointInfo(self.robot_id, i) for i in range(self.num_joints)]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        mpos = [state[0] for state in joint_states]
        mvel = [state[1] for state in joint_states]
        result = self.bc.getLinkState(self.robot_id, self.grasptarget_index, computeLinkVelocity=1, computeForwardKinematics=1)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
        # Get the Jacobians for the CoM of the end-effector link.
        # Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn.
        # The localPosition is always defined in terms of the link frame coordinates.
        zero_acceleration = [0.0] * len(mpos)
        # jac_t, 3*9, jack_r: 3*9, 9: 0-6, movable joints, 7-8: finger joints
        # the correct solutions should be exclude finger joints in the robot urdf definition.
        jac_t, jac_r = self.bc.calculateJacobian(self.robot_id, self.grasptarget_index, com_trn, mpos, mvel, zero_acceleration)
        jacobian = np.concatenate((np.array(jac_t)[:, 0:7], np.array(jac_r)[:, 0:7]), axis=0)
        return jacobian  # 6*7

