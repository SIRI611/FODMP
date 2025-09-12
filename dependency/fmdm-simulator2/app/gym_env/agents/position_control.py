#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
from utils import *


MAX_JOINT_VELs = [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]
MAX_CART_VELs = [0.008, 0.008, 0.008, 0.02]

def clamp_action(action):
    for i in range(action.shape[0]):
        action[i] = min(
            max(action[i], -MAX_JOINT_VELs[i]), MAX_JOINT_VELs[i]
        )
    return action


class ExpertAgent:
    def __init__(self):
        self.nnnn = 5

    def act(self, state, ref_state, robot_model, bc, add_noise=True):
        # state = {
        #     'joint_states': joint_state_vector,
        #     'ee_states': ee_state_vector,
        #     'image': image,
        #     'depth': depth
        # }
        # quaternion vector from pybullet:  [x,y,z,qx,qy,qz,qw]
        # quaternion vector required in pytransform3d: (x, y, z, qw, qx, qy, qz)
        ct = pbvs6(state['ee_states'], ref_state['ee_states'])
        # now map ct to action through jacobian.
        jacobian = robot_model.get_jacobian()  # 6*7 matrix

        # jacobian = jacobian[3:6, :]
        # ct = ct[0:3]
        joint_vel = np.matmul(np.linalg.pinv(jacobian), np.expand_dims(ct, axis=1))
        # print(joint_vel)
        action = np.squeeze(joint_vel)  # [random.uniform(-max_vel * 0.2, max_vel * 0.2) for max_vel in MAX_JOINT_VELs]
        # action = np.array([ct[0], ct[1], ct[2], ct[5]])*0.1
        print(action)
        sigma = 0.1
        if add_noise:
            action += np.random.normal(0, sigma, len(action))
        # add clamp to fix max vels
        return clamp_action(action)


class RandomAgent:
    def __init__(self, ref_state, bc):
        self.random_target = self.initialize_random_target(ref_state, bc)

    def initialize_random_target(self, ref_state, bc):
        # uniformly sample from ref_state, x [-0.2, 0.2], y[-0.2, 0.2], z[0, 0.3]
        # rotation, roll, pitch, yaw [-70,70]
        ref_pos = np.array(ref_state['ee_states'])
        ref_pos[0:2] += np.random.uniform(-0.15, 0.15, 2)
        ref_pos[2] += np.random.uniform(0, 0.2, 1)[0]
        rotation_angle = np.radians(10)
        eulers = np.array(bc.getEulerFromQuaternion(ref_pos[3:7])) + np.random.uniform(-rotation_angle, rotation_angle, 3)
        ref_pos[3:7] = np.array(bc.getQuaternionFromEuler(eulers))
        return ref_pos


    def act(self, state, ref_state, robot_model, bc, add_noise=True):
        ct = pbvs6(state['ee_states'], self.random_target)
        # now map ct to action through jacobian.
        jacobian = robot_model.get_jacobian()  # 6*7 matrix
        # jacobian = jacobian[3:6, :]
        # ct = ct[0:3]
        joint_vel = np.matmul(np.linalg.pinv(jacobian), np.expand_dims(ct, axis=1))
        # print(joint_vel)
        action = np.squeeze(joint_vel)*0.5  # [random.uniform(-max_vel * 0.2, max_vel * 0.2) for max_vel in MAX_JOINT_VELs]
        sigma = 0.1
        if add_noise:
            action += np.random.normal(0, sigma, len(action))*0.5
        # add clamp to fix max vels
        return clamp_action(action)
