import numpy as np
import math
import os

MOVING_BASE = 0
MOVING_FINGERS = 1


class CartesianRobot:
    def __init__(
        self,
        configs,
        bullet_client,
        default_finger_distance_ratio=1.0,
        verbose=False
    ):
        """
        Initializes a cartesian robot.
        
        Parameters
        ----------
        configs : dict
            Dictionary consisting of configuration to load. 
        bullet_client : pybullet_utils.bullet_client.BulletClient
            Pybullet client.
        default_finger_distance_ratio : float 
            distance = ratio * max_finger_distance
        verbose : boolean
            Whether increased verbosity is to be used in observations, etc.
        """
        self.bin_id, self.constraint_id, self.robot_id = None, None, None
        self.verbose = verbose
        self.enable_gripper_force_readings = configs['enable-gripper-force-readings']
        self.linear_damping = configs['linear-damping']
        self.angular_damping = configs['angular-damping']
        self.base_pos = configs["position"]
        self.base_orientation = configs["orientation"]
        self.gear_settings = configs['gear']
        self.gripper_settings = configs['gripper']
        self.retract_action = configs['retract-action']
        self.place_down_settings = configs['place-down']
        self.default_d_ratio = default_finger_distance_ratio
        self.bc = bullet_client
        self.flags = (
            self.bc.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | 
            self.bc.URDF_USE_SELF_COLLISION |
            self.bc.URDF_ENABLE_SLEEPING |
            self.bc.URDF_MAINTAIN_LINK_ORDER
        )
        self.joint_indices = [0, 1]
        self.urdf_file = os.environ['path-to-robots'] + 'cartesian/cartesian.urdf'
        self.reset()

    def apply_action(self, pos, yaw, d, _bin_id=-1):
        """
        The apply action process carries out the following steps in the given order.
        (0) back to reset pose
        (1) move to pre-grasp pose
        (2) grasp 
        (3) retract to decide successful grasp or not 
        (4) remove obj or place down obj
        
        Parameters
        ----------
        pos : list
            Target position in the form of [x, y, z], list of length 3.
        yaw : float
            Target orientation of cartesian robot, angle phi in range [-pi/2, pi/2].
        d : float
            Finger distance.
        _bin_id : int
            Unique id of the bin filled with objects for collision detection.
        Returns
        -------
        int
            -1 if the robot has collided with the bin, 0 otherwise.
        """
        rxn_forces_index = 1
        self.reset()
        self.bin_id = _bin_id
        if self.verbose:
            print('      ', end='')
        status_of_finger_action = self.finger_control(d, max_force=100, task_name='open finger')
        if status_of_finger_action is -1:
            return -1
        elif self.enable_gripper_force_readings:
            self.print_reaction_forces(
                msg='      reaction forces during finger action:',
                rxn_forces_dict=status_of_finger_action[rxn_forces_index]
            )
        status_of_gripper_action = self.gripper_control(pos, yaw, max_force=None, task_name='pre-grasp')
        if status_of_gripper_action is -1:
            return -1
        elif self.enable_gripper_force_readings:
            self.print_reaction_forces(
                msg='      reaction forces during gripper action:',
                rxn_forces_dict=status_of_gripper_action[rxn_forces_index]
            )
        status_of_close_gripper = self.close_gripper()
        if self.enable_gripper_force_readings:
            self.print_reaction_forces(
                msg='      reaction forces while closing gripper:',
                rxn_forces_dict=status_of_finger_action[rxn_forces_index]
            )
        # self.retract()
        if self.verbose:
            print('\n')
        return 0

    def apply_action_diff(self, action, height_threshold, _bin_id=-1):
        """
        Applies a delta action as defined in https://arxiv.org/pdf/1802.10264.pdf
        The self.retract() method call decides if a successful grasp has occured or not.
        Comment out the line below that calls the method if you don't need enforce this stable
        grasp strictness.

        Parameters
        ----------
        action : list
            Consists of the desired change in x, y, z, and phi to apply to the cartesian robot.
            In the form [dx,dy,dz,d_phi], list of length 4.
        height_threshold : float
            The gripper automatically closes when it moves below a fixed height threshold, and
            the episode ends. Measured in world coordinate frame.
        _bin_id : int
            Unique id of bin for collision detection.
        Returns
        -------
        int
            -2 if a grasp attempt tried, -1 if a collision occured, 0 if not.
        """
        rxn_forces_index = 1
        dx, dy, dz, d_phi = action
        robot_state = self.get_robot_state()
        pos = np.array(robot_state[0:3]) + np.array([dx, dy, dz])
        yaw = robot_state[3] + d_phi
        self.bin_id = _bin_id
        if self.verbose: 
            print('      ', end='')
        status_of_gripper_action = self.gripper_control(pos, yaw, max_force=None, task_name='pre-grasp')
        if status_of_gripper_action is -1:
            return -1
        elif self.enable_gripper_force_readings:
            self.print_reaction_forces(
                msg='      reaction forces during gripper action:',
                rxn_forces_dict=status_of_gripper_action[rxn_forces_index]
            )
        if pos[2] <= height_threshold:
            status_of_close_gripper = self.close_gripper()
            status_of_gripper = self.retract()
            if self.enable_gripper_force_readings:
                self.print_reaction_forces(
                    msg='      reaction forces while closing gripper:',
                    rxn_forces_dict=status_of_close_gripper[rxn_forces_index]
                )
                self.print_reaction_forces(
                    msg='      reaction forces while retracting gripper:',
                    rxn_forces_dict=status_of_gripper[rxn_forces_index]
                )
            return -2
        if self.verbose:
            print('\n')
        return 0

    def avoid_camera(self):
        """
        Sets the position of the robot so it is out of the camera's field of view.
        """
        self.bc.resetBasePositionAndOrientation(
            self.robot_id,
            [self.base_pos[0], self.base_pos[1]-0.3, self.base_pos[2]],
            self.base_orientation
        )

    def back_to_initial(self):
        """
        Sets the position of the robot back to its initial position.
        """
        self.bc.resetBasePositionAndOrientation(
            self.robot_id,
            [self.base_pos[0], self.base_pos[1], self.base_pos[2]],
            self.base_orientation
        )

    def close_gripper(self):
        """
        Closes the gripper of the cartesian robot.

        Returns
        -------
        tuple
            An int and a dict; the int corresponds to the number of times that the pybullet physics engine had to be
            stepped in order for the robot to reach its destination and the dict corresponds to the end effector force
            sensor readings at each pybullet physics engine step it took to reach the destination (will be an empty
            dict if the get-end-effector-force-sensor-readings option is disabled in the config file). 
        """
        return self.finger_control(0, None, task_name='finger close')

    def enable_force_sensor_on_fingers(self):
        """
        Enables a force/torque sensor in both of the robot's end effectors (fingers).
        """
        a, b = 0, 1
        self.bc.enableJointForceTorqueSensor(self.robot_id, self.joint_indices[a], enableSensor=1)
        self.bc.enableJointForceTorqueSensor(self.robot_id, self.joint_indices[b], enableSensor=1)

    def finger_control(self, distance, max_force=None, task_name=''):
        """
        Controls movements of the cartesian robot's end effectors (fingers).

        Parameters
        ----------
        distance : float
            Distance to move both fingers.
        max_force : float
            Maximum force to use while closing end effectors.
        task_name : string
            Used for debugging if verbose is enabled.
        Returns
        -------
        tuple
            An int and a dict; the int corresponds to the number of times that the pybullet physics engine had to be
            stepped in order for the robot to reach its destination and the dict corresponds to the end effector force
            sensor readings at each pybullet physics engine step it took to reach the destination (will be an empty
            dict if the get-end-effector-force-sensor-readings option is disabled in the config file). 
        """
        assert (distance <= self.gripper_settings['maximum-distance']), \
            'distance {} > max-distance {}'.format(distance, self.gripper_settings['maximum-distance'])
        joint_poss = np.ones(len(self.joint_indices)) * distance
        if max_force is None:
            max_force = self.gripper_settings['gripping-force']
        forces = np.ones(len(self.joint_indices)) * max_force
        self.bc.setJointMotorControlArray(
            self.robot_id,
            jointIndices=self.joint_indices,
            controlMode=self.bc.POSITION_CONTROL,
            targetPositions=joint_poss,
            forces=forces
        )
        if distance is not 0:
            inner_steps, reaction_forces = self.go_to_position(1, [distance, distance])
        else:
            inner_steps, reaction_forces = self.go_to_position(1, [0, 0], max_steps=50)
        if self.verbose:
            print('=>[{}:{}]'.format(task_name, inner_steps), end='')
        return inner_steps, reaction_forces

    def get_end_effector_reaction_forces(self):
        """
        Gets the reaction forces sensed by each end effector along the z-axis of their reference frame. Currently the
        other sensed forces [Fx,Fy] and sensed torques [Mx,My,Mz] are ignored.

        Returns
        -------
        tuple
            Two floats; the first float corresponds to the reaction force sensed by end effector a (joint index 9)
            along its z-axis and the second float corresponds to the reaction forces sensed by end effector b (joint
            index 10) its z-axis.
        """
        a, b, reaction_forces_index, z = 0, 1, 2, 2
        end_effector_states = self.bc.getJointStates(self.robot_id, self.joint_indices)
        end_effector_a_rxn_force_z = end_effector_states[a][reaction_forces_index][z]
        end_effector_b_rxn_force_z = end_effector_states[b][reaction_forces_index][z]
        return end_effector_a_rxn_force_z, end_effector_b_rxn_force_z

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
        grasptarget_index = 2
        grasptarget_link_state = self.bc.getLinkState(self.robot_id, grasptarget_index)
        grasptarget_pos = np.array(grasptarget_link_state[0])
        grasptarget_orn_euler = np.array(self.bc.getEulerFromQuaternion(grasptarget_link_state[1]))
        return grasptarget_pos, grasptarget_orn_euler

    def get_robot_state(self):
        """
        Gets the current position and angle of the cartesian robot as well as the distance between the end effectors.

        Returns
        -------
        list
            The current robot state in the form of [x, y, z, yaw, d*2] (list of length 5).
        """
        pos, orientation = self.bc.getBasePositionAndOrientation(self.robot_id)
        yaw = np.array(self.bc.getEulerFromQuaternion(orientation))[2]
        j_pos1, _, _, j_f1 = self.bc.getJointState(self.robot_id, 0)
        j_pos2, _, _, j_f2 = self.bc.getJointState(self.robot_id, 1)
        return [pos[0], pos[1], pos[2], yaw, abs(j_pos2 - j_pos1)]

    def go_to_position(self, moving_mark, desired_pose, max_steps=200):
        """
        Steps the pybullet physics engine until the robot is within a certain threshold of its destination position
        up to a max of max_steps. In other words, the strategy is to call many step()s, until joint or body stops
        moving.
        
        Parameters
        ----------
        moving_mark : int
            Whether the robot is moving or the end effectors (fingers) are moving. MOVING_BASE=0. MOVING_FINGERS=1.
        desired_pose : list
            Destination pose as either the target coordinates of the gripper (list of length 3) or the target poses
            of each end effector (list of length 2).
        max_steps : int
            Maximum allowed number of times to step the simulation.
        Returns
        -------
        tuple
            An int and a dict; the int corresponds to the number of times that the pybullet physics engine had to be
            stepped in order for the robot to reach its destination and the dict corresponds to the end effector force
            sensor readings at each pybullet physics engine step it took to reach the destination (will be an empty
            dict if the get-end-effector-force-sensor-readings option is disabled in the config file). 
        """
        reaction_forces = {
            'end_effector_a_reaction_force_z': [],
            'end_effector_b_reaction_force_z': []
        }
        count = 0
        for _ in range(max_steps):
            if self.enable_gripper_force_readings:
                end_effector_a_rxn_force_z, end_effector_b_rxn_force_z = self.get_end_effector_reaction_forces()
                reaction_forces['end_effector_a_reaction_force_z'].append(end_effector_a_rxn_force_z)
                reaction_forces['end_effector_b_reaction_force_z'].append(end_effector_b_rxn_force_z)
            if self.safe_step() is False:
                return -1, reaction_forces
            count += 1
            if self.is_moving(moving_mark, desired_pose) is False:
                break
        return count, reaction_forces

    def gripper_control(self, pos, yaw=None, max_force=None, task_name=''):
        """
        Move gripper to pre-grasp pose.

        Parameters
        ----------
        pos : list
            Target gripper base position in the form [x, y, z].
        yaw : float
            Target gripper angle, phi, in the range [-pi, pi].
        max_force : float
            Maximum force the constraint attached to the grippers base can apply.
        task_name : string
            Used for debugging if verbose is enabled.
        Returns
        -------
        tuple
            An int and a dict; the int corresponds to the number of times that the pybullet physics engine had to be
            stepped in order for the robot to reach its destination and the dict corresponds to the end effector force
            sensor readings at each pybullet physics engine step it took to reach the destination (will be an empty
            dict if the get-end-effector-force-sensor-readings option is disabled in the config file).
        """
        _, orientation = self.bc.getBasePositionAndOrientation(self.robot_id)
        eulers = np.array(self.bc.getEulerFromQuaternion(orientation))
        if yaw is None:
            yaw = eulers[2]
        else:
            eulers[2] = yaw
        if max_force is None:
            self.bc.changeConstraint(self.constraint_id, pos, self.bc.getQuaternionFromEuler(eulers))
        else:
            self.bc.changeConstraint(self.constraint_id, pos, self.bc.getQuaternionFromEuler(eulers), maxForce=max_force)
        inner_steps, reaction_forces = self.go_to_position(MOVING_BASE, [pos[0], pos[1], pos[2], yaw/100], max_steps=60)
        self.bc.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
        if self.verbose:
            print('=>[{}:{}]'.format(task_name, inner_steps), end='')
        return inner_steps, reaction_forces

    def is_moving(self, moving_mark, desired_pose):
        """
        Decide if the base or the end effectors (fingers) are within a specified threshold of their destination or not.

        Parameters
        ----------
        moving_mark : int
            Whether the robot is moving or the end effectors (fingers) are moving. MOVING_BASE=0. MOVING_FINGERS=1.
        desired_pose : list
            Destination pose as either the target coordinates of the gripper (list of length 3) or the target poses
            of each end effector (list of length 2).
        Returns
        -------
        boolean
            Whether or not the base or end effectors are moving or not.
        """
        if moving_mark == MOVING_BASE:
            return self.is_moving_base(desired_pose)
        elif moving_mark == MOVING_FINGERS:
            return self.is_moving_joints(desired_pose)

    def is_moving_base(self, desired_pose, threshold=0.001):
        """
        Decide if the robot base is within a specified threshold of its destination position or not.

        Parameters
        ----------
        desired_pose : list
            The target coordinates of the gripper (list of length 3).
        threshold : float
            If the robot is within this threshold of its destination, then the action is considered successfully
            completed.  
        Returns
        -------
        boolean
            Whether or not the base is moving or not.
        """
        pos, orientation = self.bc.getBasePositionAndOrientation(self.robot_id)
        yaw = np.array(self.bc.getEulerFromQuaternion(orientation))[2]
        pose = list(pos)
        pose.append(yaw/100)
        pose = np.array(pose)
        if np.linalg.norm(pose - np.array(desired_pose)) > threshold:
            return True
        return False

    def is_moving_joints(self, desired_pose, threshold=0.002):
        """
        Decide if the end effectors (fingers) are within a specified threshold of their destination position or not.

        Parameters
        ----------
        desired_pose : list
            The target poses of each end effector (list of length 2).
        threshold : float
            If the end effectors are within this threshold of their destination, then the action is considered
            successfully completed.
        Returns
        -------
        boolean
            Whether or not the base is moving or not.
        """
        j_pos1, _, _, j_f1 = self.bc.getJointState(self.robot_id, 0)
        j_pos2, _, _, j_f2 = self.bc.getJointState(self.robot_id, 1)
        if math.sqrt((j_pos1-desired_pose[0])**2 + (j_pos2-desired_pose[1])**2) > threshold:
            return True
        return False

    def open_gripper(self, max_force=1000):
        """
        Opens the gripper of the cartesian robot.
        
        Parameters
        ----------
        max_force : float
            Maximum force that each end effector can use to open.
        Returns
        -------
        tuple
            An int and a dict; the int corresponds to the number of times that the pybullet physics engine had to be
            stepped in order for the robot to reach its destination and the dict corresponds to the end effector force
            sensor readings at each pybullet physics engine step it took to reach the destination (will be an empty
            dict if the get-end-effector-force-sensor-readings option is disabled in the config file).
        """
        return self.finger_control(self.gripper_settings["maximum-distance"], max_force=max_force, task_name='gripper open')

    def place_down(self):
        """
        Place down and release the grasped object.
        """
        self.gripper_control(
            self.place_down_settings["position"],
            None,
            max_force=self.retract_action["maximum-force"],
            task_name='place down'
        )
        self.open_gripper()
    
    def print_reaction_forces(self, msg, rxn_forces_dict):
        """
        Prints the reaction forces dictionary in a nicely formatted way.

        Parameters
        ----------
        msg : string
            Header to print - ideally the incorporating the action that was applied to yield the reaction forces.
        rxn_forces_dict : dict
            Dictionary consisting of the reaction forces felt by each end effector at each pybullet physics engine
            step. 
        """
        print('\n{}'.format(msg), end='')
        for key, values in rxn_forces_dict.items():
            print('\n         {}'.format(key), end=':  ')
            for force in values:
                print('{:.3f}'.format(force), end='  ')

    def reset(self):
        """
        Resets the cartesian robot to its default position (spawning a new robot is the faster than travelling back).
        """
        try:
            if self.robot_id is not None:
                self.bc.removeBody(self.robot_id)
            self.robot_id = self.bc.loadURDF(
                fileName=self.urdf_file,
                basePosition=[self.base_pos[0], self.base_pos[1], self.base_pos[2]],
                baseOrientation=self.base_orientation,
                flags=self.flags
            )
            if self.robot_id < 0:
                raise Exception('Cannot load URDF file.')
            else:
                c = self.bc.createConstraint(
                    parentBodyUniqueId=self.robot_id,
                    parentLinkIndex=0,
                    childBodyUniqueId=self.robot_id,
                    childLinkIndex=1,
                    jointType=self.bc.JOINT_GEAR,
                    jointAxis=[1, 0, 0],
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=[0, 0, 0]
                )
                self.bc.changeConstraint(
                    userConstraintUniqueId=c,
                    gearRatio=-1,
                    erp=self.gear_settings['erp'],
                    maxForce=self.gear_settings['maximum-force']
                )
                self.bc.changeDynamics(self.robot_id, 0, linearDamping=self.linear_damping, angularDamping=self.angular_damping)
                self.bc.changeDynamics(self.robot_id, 1, linearDamping=self.linear_damping, angularDamping=self.angular_damping)
                self.constraint_id = self.bc.createConstraint(
                    parentBodyUniqueId=self.robot_id,
                    parentLinkIndex=-1,
                    childBodyUniqueId=-1,
                    childLinkIndex=-1,
                    jointType=self.bc.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=self.base_pos,
                    parentFrameOrientation=[0, 0, 0, 1],
                    childFrameOrientation=self.base_orientation
                )
                if self.enable_gripper_force_readings:
                    self.enable_force_sensor_on_fingers()
        except Exception as ex2:
            self.bc.disconnect()
            print(type(ex2))
            print(ex2.args[0])
            exit()

    def retract(self, hold_steps=5, retract_distance=0.01):
        """
        Executes a retract action consisting of the following steps in the given order.
        1) raise up the gripper and hold in that position for several steps
        2) decide if grasp is successful
        
        Parameters
        ----------
        hold_steps : int
            Number of times the pybullet physics engine is to be stepped, during which the robot must continue holding
            the object for the grasp to be deemed a stable grasp.
        retract_distance : float
            Distance to move the robot up.
        Returns
        -------
        tuple
            An int and a dict; the int corresponds to the number of times that the pybullet physics engine had to be
            stepped in order for the robot to reach its destination and the dict corresponds to the end effector force
            sensor readings at each pybullet physics engine step it took to reach the destination (will be an empty
            dict if the get-end-effector-force-sensor-readings option is disabled in the config file).
        """
        pos, _ = self.bc.getBasePositionAndOrientation(self.robot_id)
        pos = list(pos)
        pos[2] = self.retract_action["z-distance"] + retract_distance
        inner_steps, reaction_forces = self.gripper_control(
            pos,
            None,
            max_force=self.retract_action["maximum-force"],
            task_name='retract'
        )
        pos[2] = self.retract_action["z-distance"]
        inner_steps, reaction_forces = self.gripper_control(
            pos,
            None,
            max_force=self.retract_action["maximum-force"],
            task_name='retract'
        )
        for _ in range(hold_steps):
            self.bc.stepSimulation()
        return inner_steps, reaction_forces

    def safe_step(self, distance_threshold=-0.001):
        """
        Checks if the cartesian robot is currently colliding with bin walls.
        
        Returns
        -------
        boolean
            False if the cartesian robot is currently colliding with bin walls, True if not.
        """
        self.bc.stepSimulation()
        contact_points = self.bc.getContactPoints(self.robot_id, self.bin_id)
        if len(contact_points) > 0:
            for c_p in contact_points:
                if c_p[8] < distance_threshold:
                    return False
        else:
            return True
