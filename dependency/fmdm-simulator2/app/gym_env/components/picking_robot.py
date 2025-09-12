import os
import math
import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import copy

def change_qxyzw2qwxyz(q):
    new_q = np.concatenate((q[3:4], q[0:3]), axis=0)
    return new_q

def pbvs6(current_pos, ref_pos, _lambda_xyz=1.5, _lambda_orn=1.5):
    current_xyz = np.array(current_pos['position'])
    ref_xyz = np.array(ref_pos['position']) + [0,0,0.005]
    residual_error = np.linalg.norm(ref_xyz - current_xyz)
    if residual_error<0.008:
        ref_xyz = np.array(ref_pos['position']) - [0, 0, 0.003]
    error_xyz = (ref_xyz - current_xyz)*_lambda_xyz
    # next_xyz = current_xyz + error_xyz
    current_qxyzw = change_qxyzw2qwxyz(current_pos['orientation'])
    ref_qxyzw = change_qxyzw2qwxyz(ref_pos['orientation'])
    # Compute the rotation in angle-axis format that rotates next_qxyzw into current_qxyzw.
    (x, y, z, angle) = pr.quaternion_diff(current_qxyzw, ref_qxyzw)
    error_orn = (-1) * _lambda_orn * angle * np.array([x, y, z])
    # print('current | target')
    # print('[xyz {:+.4f}] [{:+.4f}|{:+.4f}, {:+.4f}|{:+.4f}, {:+.4f}|{:+.4f}], q [{:+.4f}|{:+.4f}, {:+.4f}|{:+.4f}, {:+.4f}|{:+.4f}, {:+.4f}|{:+.4f}]'.format(residual_error, current_xyz[0], ref_xyz[0], current_xyz[1], ref_xyz[1], current_xyz[2], ref_xyz[2], current_pos[3], ref_pos[3], current_pos[4], ref_pos[4],current_pos[5], ref_pos[5],current_pos[6], ref_pos[6]))
    return np.hstack((error_xyz, error_orn)), residual_error

MAX_JOINT_VELs = [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]
MAX_CART_VELs = [0.008, 0.008, 0.008, 0.02]

def clamp_action(action):
    for i in range(action.shape[0]):
        action[i] = min(
            max(action[i], -MAX_JOINT_VELs[i]), MAX_JOINT_VELs[i]
        )
    return action

class PickingRobot:
    """
    Class providing functionality and support for a picking robot. Allows for control and interaction of the robot in a
    simulator (designed for pybullet). Note that a robot being modelled by this class should have an end effector that
    has a gripper consisting of two fingers and a "grasptarget" joint corresponding to the gripper's grasptarget (the
    location between the fingers at which an object should ideally be if it is to be successfully grasped).
    """
    max_steps = 50
    max_grasping_force = 80
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


    def go_to_ee_target_one_step(self, state, ref_state,  add_noise=False):
        ct, residual_error = pbvs6(state, ref_state)
        # now map ct to action through jacobian.
        jacobian = self.get_jacobian()  # 6*7 matrix
        # jacobian = jacobian[3:6, :]
        # ct = ct[0:3]
        joint_vel = np.matmul(np.linalg.pinv(jacobian), np.expand_dims(ct, axis=1))
        # print(joint_vel)
        action = np.squeeze(joint_vel)  # [random.uniform(-max_vel * 0.2, max_vel * 0.2) for max_vel in MAX_JOINT_VELs]
        # action = np.array([ct[0], ct[1], ct[2], ct[5]])*0.1
        # print(action)
        sigma = 0.1
        if add_noise:
            action += np.random.normal(0, sigma, len(action))
        # add clamp to fix max vels
        return clamp_action(action), residual_error

    def go_to_position(self, target_x_y_z, max_steps=50, error_thres=0.001):
        ee_states = self.get_end_effector_states()
        print('current ee state')
        target_ornt = [90,0,0]
        base_orn_euler = [math.radians(angle) for angle in target_ornt]
        base_orn_quaternion = self.bc.getQuaternionFromEuler(base_orn_euler)
        # ee_eular = self.bc.getEulerFromQuaternion(ee_states['orientation'])
        # print(ee_states)
        # print('current ee eulers')
        # print(ee_eular)
        target_ee_states = copy.deepcopy(ee_states)
        target_ee_states['position']=target_x_y_z
        target_ee_states['orientation'] = base_orn_quaternion
        for _ in range(max_steps):
            joint_vels, residual_error = self.go_to_ee_target_one_step(ee_states, target_ee_states)
            if residual_error <=error_thres:
                break
            self.apply_action_vel(list(joint_vels*0.1)[ : -1] + [0, 0,0])
            # print('residual error {}'.format(residual_error))

    def apply_action_yk(self, target):
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
        for _ in range(200):
            self.bc.stepSimulation()


    def apply_action(self, control_mode, target):
        """
        Applies action to this robot using either position or velocity control. If position control is used, will
        call self.at_target() which steps the physics engine until the given number of joints in this robot are within
        a specified proximity to the target position or until the specified max number of allowed physics engine steps
        are reached. Also if using position control, the fingers are moved with a max force of 80 and the reaction
        forces sensed by each of them are returned (only calculated if enable-gripper-force-readings is set to True
        in the config file). If velocity control is used, the physics engine is stepped once after the
        velocities have been applied to the joints and end effectors and nothing is returned.

        Parameters
        ----------
        control_mode : int
            Means of applying action to robot (self.bc.POSITION_CONTROL or self.bc.VELOCITY_CONTROL).
        target : list
            Target joint pose for each moveable joint and the gripper's fingers.
        Returns
        -------
        dict
            (Situational) In the case that positional control is used, this method will return a dictionary consisting
            of a key-value pair where each key corresponds to one of the fingers that make up the gripper and the
            value consists of a list of the reaction forces sensed by the respective finger at each simulation step
            taken for the robot to get to its destination. In the case that velocity control is used, nothing is
            returned.
        """
        positional_control = self.bc.POSITION_CONTROL
        if control_mode == positional_control:
            self.bc.setJointMotorControlArray(
                bodyIndex=self.robot_id,
                jointIndices=self.moveable_joints,
                controlMode=control_mode,
                targetPositions=target[:len(self.moveable_joints)]
            )
            self.bc.setJointMotorControlArray(
                bodyIndex=self.robot_id,
                jointIndices=self.finger_indices,
                controlMode=control_mode,
                targetPositions=target[len(self.moveable_joints):],
                forces=[self.max_grasping_force] * len(self.finger_indices)
            )
            return self.at_target(target, True)
        else:
            self.bc.setJointMotorControlArray(
                bodyIndex=self.robot_id,
                jointIndices=self.moveable_joints + self.finger_indices,
                controlMode=control_mode,
                targetVelocities=target
            )
            self.bc.stepSimulation()

    def apply_action_pos(self, target_joint_poses):
        """
        Moves the moveable joints of this robot to their specified target pose. First ensures that the desired target
        pose of each moveable joint is valid - if so the original target pose is used, else the closest valid pose is
        used. Returns the reaction forces sensed by each finger on the gripper if enable-gripper-force-readings
        is enabled in the config file.

        Parameters
        ----------
        target_joint_poses : list
            Target joint pose for each moveable joint and the gripper's fingers.
        Returns
        -------
        any
            If enable-gripper-force-readings is enabled, this method returns a dictionary consisting of a key, value
            pair where each key corresponds to one of the gripper's fingers and the value consists of a list of the
            reaction forces sensed by the respective finger at each simulation step. Otherwise, nothing is returned.
        """
        assert len(target_joint_poses) == len(self.moveable_joints + self.finger_indices), \
            'The parameter \'target_joint_poses\' is invalid, ensure that you have specified a target velocity ' \
            'for all moveable joints and the gripper\'s fingers.'
        for i in range(len(target_joint_poses)):
            target_joint_poses[i] = min(max(target_joint_poses[i], self.lower_limits[i]), self.upper_limits[i])
        if self.enable_gripper_force_readings:
            return self.apply_action(self.bc.POSITION_CONTROL, target_joint_poses)
        else:
            self.apply_action(self.bc.POSITION_CONTROL, target_joint_poses)

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

    def apply_action_vel(self, target_joint_velocities):
        """
        Moves the moveable joints of this robot using velocity control. First checks if the desired target velocity of
        each moveable joint is valid - if so the original target velocity is used, else the closest valid velocity is used.

        Parameters
        ----------
        target_joint_velocities : list
            Target velocity for each moveable joint and the gripper's fingers.
        """
        assert len(target_joint_velocities) == len(self.moveable_joints + self.finger_indices), \
            'The parameter \'target_joint_velocities\' is invalid, ensure that you have specified a target velocity ' \
            'for all moveable joints and the gripper\'s fingers.'
        for i in range(len(target_joint_velocities)):
            target_joint_velocities[i] = min(
                max(target_joint_velocities[i], -self.max_velocities[i]), self.max_velocities[i]
            )
        self.apply_action(self.bc.VELOCITY_CONTROL, target_joint_velocities)
        INNER_STEPS = 20
        for _ in range(INNER_STEPS):
            # close gripper. Can't use the built in function.
            # use maximum gripper force to make sure obj will not fall down.
            max_force = 80
            self.bc.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.finger_indices,
                controlMode=self.bc.POSITION_CONTROL,
                targetPositions=[0, 0],
                forces=[max_force, max_force]
            )
            self.bc.stepSimulation()

    def apply_action_vel_impedance(self, target_joint_velocities):
        """
        Moves the moveable joints of this robot using internal impedance velocity control. First checks if the desired target velocity of
        each moveable joint is valid - if so the original target velocity is used, else the closest valid velocity is used.
        Calculates the impeded velocity using: v = v_input - sigma * gravity_compensated_force for a number of inner steps.
        using internal impedance control to apply joint velocities.
        ** since we assume the object is light-weight, gravity_compensation is 0 now.

        If sigma and inner steps are configured in config:robot:impedance then this will behave like
        apply_action_vel just without the finger velocities.

        Parameters
        ----------
        target_joint_velocities : list
            Target velocity for each moveable joint.
        """
        assert len(target_joint_velocities) == len(self.moveable_joints), \
            'The parameter \'target_joint_velocities\' is invalid, ensure that you have specified a target velocity ' \
            'for only the moveable joints'
        for _ in range(self.inner_steps):
            target_joint_velocities_impeded = []
            joint_torques = [joint_info[3] for joint_info in
                             self.bc.getJointStates(self.robot_id, self.moveable_joints)]
            # print(max(map(abs,joint_torques)))
            for target_joint_velocity, joint_torque, sigma in zip(target_joint_velocities, joint_torques, self.sigma):
                target_joint_velocities_impeded.append(
                    target_joint_velocity - sigma * joint_torque
                )
            print(target_joint_velocities_impeded)
            target_joint_velocities_impeded.extend((0, 0))
            self.apply_action_vel(target_joint_velocities_impeded)
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
            for _ in range(self.delay_steps):
                self.bc.stepSimulation()

    def at_target(self, target, check_fingers):
        """
        Steps the physics engine until either the poses of the moveable joints and the gripper's fingers (depending on
        check_fingers) are within an acceptable proximity of their target poses or self.max_steps is reached. Returns
        the reaction forces sensed by each one of the gripper's fingers along their z-axis (only calculates them if
        enable-gripper-force-readings is enabled in the config file).

        Parameters
        ----------
        target : list
            Target velocity for each moveable joint and the gripper's fingers.
        check_fingers : bool
            Whether to check if the gripper's fingers are within an acceptable proximity of their target pose or not.
        Returns
        -------
        dict
            A dictionary consisting of a key, value pair where each key corresponds to one of the gripper's fingers and
            the value consists of a list of the reaction forces sensed by the respective finger at each simulation
            step.
        """
        reaction_forces = {
            'end_effector_a_reaction_force_z': [],
            'end_effector_b_reaction_force_z': []
        }
        for i in range(self.max_steps):
            self.bc.stepSimulation()
            if check_fingers:
                diff = abs(target - self.get_joint_poses())
            else:
                diff = abs(target[:len(self.moveable_joints)] - self.get_joint_poses()[:len(self.moveable_joints)])
            if self.enable_gripper_force_readings:
                end_effector_a_rxn_force_z, end_effector_b_rxn_force_z = self.get_end_effector_reaction_forces()
                reaction_forces['end_effector_a_reaction_force_z'].append(end_effector_a_rxn_force_z)
                reaction_forces['end_effector_b_reaction_force_z'].append(end_effector_b_rxn_force_z)
            if np.array([angle_difference < self.acceptable_joint_proximity for angle_difference in diff]).all():
                # print('Target position successfully reached (within acceptable proximity).')
                break
        return reaction_forces

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

    def close_gripper(self):
        """
        Closes the gripper on the robot and updates the gripper's state.
        """
        closed_position = 0
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

    def get_grasptarget_state(self):
        """
        Gets the current position and orientation of the grasptarget (has same orientation as gripper).

        Returns
        -------
        tuple
            Two numpy arrays, the first consisting of coordinates of the current position of the gripper,
            shape (3,) and the second consisting of the current orientation of the gripper in euler form,
            shape (3,).
        """
        position_index, orientation_index = 0, 1
        grasptarget_link_state = self.bc.getLinkState(self.robot_id, self.grasptarget_index, computeLinkVelocity=True)
        grasptarget_pos = np.array(grasptarget_link_state[position_index])
        grasptarget_orn_euler = np.array(self.bc.getEulerFromQuaternion(grasptarget_link_state[orientation_index]))
        return grasptarget_pos, grasptarget_orn_euler, grasptarget_link_state[orientation_index]

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

    def open_gripper(self, target_position=0.02):
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
        print('gripper open command')
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

    def shake(self):
        """
        Shakes the robot by moving the head up a small distance and then back down a small distance to the original
        position. Use of this function is to validate a successful firm grab by shaking the arm and then ensuring that
        the object is still being held by the arm.
        """
        offset = 0.1
        closed_gripper = [0, 0]
        grasptarget_state = self.bc.getLinkState(bodyUniqueId=self.robot_id, linkIndex=self.grasptarget_index)
        original_position, original_orn = list(grasptarget_state[0]), list(grasptarget_state[1])
        offset_position = [original_position[0], original_position[1], original_position[2] + offset]
        original_pose = self.use_inverse_kinematics(
            grasptarget_pos=original_position,
            grasptarget_orn=original_orn,
            use_null_space=True
        )
        original_pose[len(self.moveable_joints):] = closed_gripper
        offset_pose = self.use_inverse_kinematics(
            grasptarget_pos=offset_position,
            grasptarget_orn=original_orn,
            use_null_space=True
        )
        offset_pose[len(self.moveable_joints):] = closed_gripper
        shaking_motion = [offset_pose, original_pose]
        for pose in shaking_motion:
            self.bc.setJointMotorControlArray(
                bodyIndex=self.robot_id,
                jointIndices=self.moveable_joints + self.finger_indices,
                controlMode=self.bc.POSITION_CONTROL,
                targetPositions=pose,
                forces=[self.max_grasping_force] * len(pose)
            )
            self.at_target(pose, False)

    def use_inverse_kinematics(self, grasptarget_pos, grasptarget_orn=None, use_null_space=False):
        """
        Computes the joint angles that makes the grasptarget reach a specified target position with a specified target
        orientation (optional) using inverse kinematics. Optional null-space support is available (preferred).

        Parameters
        ----------
        grasptarget_pos : list
            Target position of the robot's grasptarget in the form [x, y, z], length 3.
        grasptarget_orn : list
            Target orientation of the robot's grasptarget (gripper) in the form [x, y, z, w] (quaternion), length 4.
        use_null_space : bool
            Determines whether the specified joint limits and rest poses - as found in null_space_limits(...) are
            accounted for in the inverse kinematics calculation.
        Returns
        -------
        numpy.ndarray
            Array consisting of the target joint poses for all moveable joints in the robot as calculated by inverse
            kinematics, length equal to the number of moveable joints in the robot.
        """
        if use_null_space:
            if grasptarget_orn is not None:
                joint_poses = self.bc.calculateInverseKinematics(
                    bodyUniqueId=self.robot_id,
                    endEffectorLinkIndex=self.grasptarget_index,
                    targetPosition=grasptarget_pos,
                    targetOrientation=grasptarget_orn,
                    lowerLimits=self.lower_limits,
                    upperLimits=self.upper_limits,
                    jointRanges=self.joint_ranges,
                    restPoses=self.rest_poses,
                    maxNumIterations=200,
                    residualThreshold=0.0001
                )
            else:
                joint_poses = self.bc.calculateInverseKinematics(
                    bodyUniqueId=self.robot_id,
                    endEffectorLinkIndex=self.grasptarget_index,
                    targetPosition=grasptarget_pos,
                    lowerLimits=self.lower_limits,
                    upperLimits=self.upper_limits,
                    jointRanges=self.joint_ranges,
                    restPoses=self.rest_poses,
                    maxNumIterations=200,
                    residualThreshold=0.0001
                )
        else:
            if grasptarget_orn is not None:
                joint_poses = self.bc.calculateInverseKinematics(
                    bodyUniqueId=self.robot_id,
                    endEffectorLinkIndex=self.grasptarget_index,
                    targetPosition=grasptarget_pos,
                    targetOrientation=grasptarget_orn
                )
            else:
                joint_poses = self.bc.calculateInverseKinematics(
                    bodyUniqueId=self.robot_id,
                    endEffectorLinkIndex=self.grasptarget_index,
                    targetPosition=grasptarget_pos
                )
        return np.array(joint_poses)

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

    def apply_action_delta(self, delta_action):
        dx, dy, dz, da = delta_action
        current_gripper_pos, current_gripper_orn, _ = self.get_gripper_state()
        # current_gripper_orn[:2] = self.initial_orn[:2]
        desired_orn_euler = [0,180,20] #current_gripper_orn + np.array([0, 0, da])
        desired_orn_euler = [math.radians(angle) for angle in desired_orn_euler]
        desired_gripper_orn = self.bc.getQuaternionFromEuler(desired_orn_euler)
        desired_gripper_pos = current_gripper_pos + [dx, dy, dz]
        # desired_gripper_pos = np.clip(desired_gripper_pos, self.workspace.min, self.workspace.max)
        desired_joint_poses = self.get_joint_poses_ik(
            target_ee_position=desired_gripper_pos,
            target_ee_orientation=desired_gripper_orn,
        )
        self.apply_action_yk(desired_joint_poses)

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
