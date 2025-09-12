#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Jun Jin
# Created Date: Thu May 14 MDT 2020
# Revised Date: N/A
# =============================================================================
"""
utils functions
Copyright (c) 2019, Huawei Canada Inc.
All rights reserved.
"""

import os
import numpy as np
import collections
import math
import random
from collections import namedtuple
import torch
from PIL import Image
import pickle
import torch.utils.data as Data
from torchvision import datasets, models, transforms
import time
import yaml
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt


def configure(config_file):
    """
    Opens and reads the config file and stores the data in dictionary form as an instance attribute.

    :param config_file: (.yaml) file consisting of simulation config
    """
    with open(config_file) as config:
        try:
            return yaml.safe_load(config)
        except Exception as err:
            print('Error Configuration File:{}'.format(err))
            raise err


def numpy_img_normalize(obs):
    obs = obs.transpose((2, 0, 1)) / 255  # transform to C X H X W for pytorch support
    obs += -(np.min(obs))
    obs_max = np.max(obs)
    if obs_max != 0:
        obs /= np.max(obs) / 2
    obs += -1
    return obs


def numpy_normalize(obs):
    obs += -(np.min(obs))
    obs_max = np.max(obs)
    if obs_max != 0:
        obs /= np.max(obs) / 2
    obs += -1
    return obs

def numpy_img_to_torch(obs):
    return obs.transpose((2, 0, 1)) / 255


def grey_scale_img_norm(grey_scale_img):
    obs = grey_scale_img/255
    obs += -(np.min(obs))
    obs_max = np.max(obs)
    if obs_max != 0:
        obs /= np.max(obs) / 2
    obs += -1
    return np.expand_dims(obs,0)


def generate_random_target_robot_pose(bullet_client, slot_id, yaw_bound=[-math.pi, math.pi]):
    slot_pos, _ = bullet_client.getBasePositionAndOrientation(slot_id)
    x, y, z = slot_pos
    z = 0.9
    orn_euler = [math.pi, 0, random.uniform(yaw_bound[0], yaw_bound[1])]
    upright_orientation = bullet_client.getQuaternionFromEuler(orn_euler)
    bullet_client.addUserDebugLine([x, y, z - 0.5], [x, y, z], lineColorRGB=[1, 0, 0])
    return [x, y, z], upright_orientation


def create_a_peg(peg_slot_config, bullet_client, robot):
    default_obj_mass = 0.01
    peg_rgba_color = [0, 0.7, 0, 1]
    peg_multi_body_flags = bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | bullet_client.URDF_ENABLE_SLEEPING
    peg_file = os.environ['path-to-pegs'] + peg_slot_config['geometry'] + '.obj'
    peg_collision_shape = bullet_client.createCollisionShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=peg_file,
        meshScale=peg_slot_config['scale-factor']
    )
    peg_visual_shape = bullet_client.createVisualShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=peg_file,
        meshScale=peg_slot_config['scale-factor'],
        rgbaColor=peg_rgba_color
    )
    if peg_slot_config['peg']['spawn-in-hand']:
        finger_states = bullet_client.getLinkStates(robot.robot_id, robot.finger_indices)
        peg_position = [sum([finger_state[0][i] for finger_state in finger_states]) / 2 for i in range(3)]
        peg_position[2] -= 0.04
        peg_orientation_quat = [sum([finger_state[1][i] for finger_state in finger_states]) / 2 for i in range(4)]
    else:
        peg_position = peg_slot_config['peg']['position']
        peg_orientation_quat = bullet_client.getQuaternionFromEuler(
            [math.radians(angle) for angle in peg_slot_config['peg']['orientation']]
        )
    peg_id = bullet_client.createMultiBody(
        baseMass=default_obj_mass,
        baseCollisionShapeIndex=peg_collision_shape,
        baseVisualShapeIndex=peg_visual_shape,
        flags=peg_multi_body_flags,
        basePosition=peg_position,
        baseOrientation=peg_orientation_quat
    )
    return peg_id


def reset_robot_pose(robot, joints):
    for joint_num in range(len(robot.moveable_joints)):
        robot.bc.resetJointState(
            bodyUniqueId=robot.robot_id,
            jointIndex=robot.moveable_joints[joint_num],
            targetValue=joints[joint_num],
            targetVelocity=0
        )
    # robot.open_gripper()


def reset_robot_and_peg_in_hand(env, xy_bounds=[0.1, 0.1], yaw_bound=[-math.pi * 5 / 6, math.pi / 3]):
    env.bullet_client.removeBody(env.peg)
    x = random.uniform(-xy_bounds[0], xy_bounds[1])
    y = random.uniform(-xy_bounds[0], xy_bounds[1])
    slot_pos, slot_orn = env.bullet_client.getBasePositionAndOrientation(env.slot)
    z = slot_pos[2] + 0.18
    # print(z)
    orn_euler = [math.pi, 0, random.uniform(yaw_bound[0], yaw_bound[1])]
    # print(orn_euler[2]*180/math.pi)
    upright_orientation = env.bullet_client.getQuaternionFromEuler(orn_euler)
    joint_poses = env.franka_robot.use_inverse_kinematics([x, y, z], upright_orientation, True)
    joint_poses[7:] = [0.02, 0.02]
    reset_robot_pose(env.franka_robot, joint_poses)
    # remove peg --> reset robot (random yaw) --> 7 joints + [0.02,0.02] --> reset peg -->
    peg_slot_config = env.config_file['scene']['peg-slot']
    env.peg = create_a_peg(peg_slot_config, env.bullet_client, env.franka_robot)  # create a peg according to robot pos
    if peg_slot_config['peg']['spawn-in-hand']:
        env.franka_robot.open_gripper(0.005)
        env.franka_robot.close_gripper()
    assert env.peg > -1, 'Multi body of peg could not be created'
    # get current peg pos, reset slot pose
    peg_pos, _ = env.bullet_client.getBasePositionAndOrientation(env.peg)

    env.bullet_client.resetBasePositionAndOrientation(env.slot, [peg_pos[0], peg_pos[1], slot_pos[2]], slot_orn)
    env.bullet_client.addUserDebugLine([peg_pos[0], peg_pos[1], slot_pos[2]],
                                       [peg_pos[0], peg_pos[1], slot_pos[2] + 1.0], lineColorRGB=[1, 0, 0])
    env.bullet_client.stepSimulation()


def load_pickle_samples(pickle_file_path):
    data = pickle.load(open(pickle_file_path, "rb"))
    episode_reference = data['episode_reference']
    observations = data['observations']
    transitions = data['transitions']
    n_episodes = len(episode_reference)
    episode_reference[0]['image']
    # transition = {'episode': i_episode, 'ref_ee_pos': ref_ee_pos, 'step': i_step, 'state_idx': obs_index,
    #               'action': action, 'next_state_idx': obs_index + 1, 'behavior _policy': conf_exp['agent'],
    #               'done': done}
    # observation = {
    #     'joint_states': joint_state_vector,
    #     'ee_states': ee_state_vector,
    #     'image': image,
    #     'depth': depth
    # }


def generate_skew_mat(v):
    """
    Returns the corresponding skew symmetric matrix from a 3-vector
    """
    skew_matrix=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return skew_matrix

def pbvs(current_pos, ref_pos, _lambda=0.1):
    """

    :param current_pos: from pybullet, [x, y, z, qx,qy,qz,qw]
    :param ref_pos: from pybullet, [x, y, z, qx,qy,qz,qw]
    :return:
    """
    current_pos = np.concatenate((current_pos[0:3], current_pos[6:7], current_pos[3:6]), axis=0)
    tf_current_pos = pt.transform_from_pq(current_pos)
    ref_pos = np.concatenate((ref_pos[0:3], ref_pos[6:7], ref_pos[3:6]), axis=0)
    tf_ref_pos = pt.transform_from_pq(ref_pos)
    tf_ref_pos_inv = pt.invert_transform(tf_ref_pos)
    tf_current2ref = pt.concat(tf_ref_pos_inv, tf_current_pos)  # current to world + world to ref = current to ref
    rotation_current2ref = tf_current2ref[0:3, 0:3]
    translation_current2ref = tf_current2ref[0:3, 3]
    # (x, y, z, angle). The angle is constrained to [0, pi].
    axis_angle = pr.axis_angle_from_matrix(rotation_current2ref)
    linear_vel = np.matmul(rotation_current2ref.T, translation_current2ref)
    angular_vel = axis_angle[3] * axis_angle[0:3]
    error_xyz = np.array(ref_pos[0:3]) - np.array(current_pos[0:3])
    print(np.squeeze(error_xyz))
    return (-1) * _lambda * np.hstack((linear_vel, angular_vel))


def pbvs2(current_pos, ref_pos, _lambda=0.1):
    current_pos = np.concatenate((current_pos[0:3], current_pos[6:7], current_pos[3:6]), axis=0)
    print('current pos')
    print(current_pos[0:3])
    tf_current_pos = pt.transform_from_pq(current_pos)
    ref_pos = np.concatenate((ref_pos[0:3], ref_pos[6:7], ref_pos[3:6]), axis=0)
    # print('ref pos')
    # print(ref_pos)
    tf_ref_pos = pt.transform_from_pq(ref_pos)
    R_rotated = np.dot(tf_current_pos[0:3, 0:3].T, tf_ref_pos[0:3, 0:3])
    axis_angle = pr.axis_angle_from_matrix(R_rotated)
    theta, u = axis_angle[3], axis_angle[0:3]
    feature_current = np.concatenate((tf_current_pos[0:3, 3], theta * u), axis=0)
    feature_target = np.concatenate((tf_ref_pos[0:3, 3], np.zeros(3)), axis=0)
    error = feature_current - feature_target
    # print(np.squeeze(error))

    # interaction matrix
    axis_angle = pr.axis_angle_from_matrix(tf_current_pos[0:3, 0:3])
    theta, u = axis_angle[3], axis_angle[0:3]
    L_top = np.concatenate((-np.identity(3), generate_skew_mat(tf_ref_pos[0:3, 3])), axis=1)
    L_bottom = np.concatenate((np.zeros((3, 3)), (np.identity(3) - theta / 2 * generate_skew_mat(u) + (
                1 - (np.sinc(theta) / (np.sinc(theta / 2) * np.sinc(theta / 2)))) * np.dot(generate_skew_mat(u),
                                                                                           generate_skew_mat(u)))),
                              axis=1)

    L = np.concatenate((L_top, L_bottom), axis=0)
    vel = -_lambda * np.dot(np.linalg.pinv(L), error)
    next_pos = np.array(current_pos[0:3]) + vel[0:3]
    print('next pos')
    print(next_pos)
    return vel


def go_to_position(pos, ortn, env):
    """

    :param pos:
    :param ortn:  [x,y,z,w]
    :param env:
    :return:
    """
    joint_poses = env.franka_robot.use_inverse_kinematics(pos, ortn, True)
    joint_poses[7:] = [0, 0]
    reset_robot_pose(env.franka_robot, joint_poses)

def expert_vel(current_pos, ref_pos, env, _lambda=0.1):
    current_xyz = np.array(current_pos[0:3])
    current_eulers = np.array(env.bullet_client.getEulerFromQuaternion(current_pos[3:7]))
    ref_xyz = np.array(ref_pos[0:3])
    next_xyz = current_xyz + (ref_xyz - current_xyz) * _lambda
    ref_eulers = np.array(env.bullet_client.getEulerFromQuaternion(ref_pos[3:7]))
    next_eulers = current_eulers + (ref_eulers - current_eulers) * _lambda
    next_q_xyzw = env.bullet_client.getQuaternionFromEuler(next_eulers)
    # current_q = np.concatenate((current_pos[6:7], current_pos[3:6]), axis=0)
    #
    # ref_q = np.concatenate((ref_pos[6:7], ref_pos[3:6]), axis=0)
    # # w, x, y, z
    # next_q_wxyz = pr.quaternion_slerp(current_q, ref_q, _lambda)
    # next_q_xyzw = np.concatenate((ref_pos[1:4], ref_pos[0:1]), axis=0)
    go_to_position(next_xyz, next_q_xyzw, env)


def pbvs4(current_pos, ref_pos, _lambda=0.1):
    """

    :param current_pos: from pybullet, [x, y, z, qx,qy,qz,qw]
    :param ref_pos: from pybullet, [x, y, z, qx,qy,qz,qw]
    :return:
    """
    current_xyz = np.array(current_pos[0:3])
    ref_xyz = np.array(ref_pos[0:3])
    error = (ref_xyz - current_xyz) * _lambda
    next_xyz = current_xyz + error
    residual_error = np.linalg.norm(ref_xyz - current_xyz)
    print('current | target | error | next')
    print('[{:+.4f}] [{:+.4f}|{:+.4f}|{:+.4f}|{:+.4f}, {:+.4f}|{:+.4f}|{:+.4f}|{:+.4f}, {:+.4f}|{:+.4f}|{:+.4f}|{:+.4f}]'.format(residual_error, current_xyz[0], ref_xyz[0], error[0], next_xyz[0], current_xyz[1], ref_xyz[1], error[1], next_xyz[1], current_xyz[2], ref_xyz[2], error[2], next_xyz[2]))
    return error

def change_qxyzw2qwxyz(q):
    new_q = np.concatenate((q[3:4], q[0:3]), axis=0)
    return new_q

def pbvs5(current_pos, ref_pos, _lambda=0.1):
    current_qxyzw = change_qxyzw2qwxyz(current_pos[3:7])
    ref_qxyzw = change_qxyzw2qwxyz(ref_pos[3:7])
    next_qxyzw = pr.quaternion_slerp(current_qxyzw, ref_qxyzw, _lambda)
    #  (w, x, y, z)
    # Compute the rotation in angle-axis format that rotates next_qxyzw into current_qxyzw.
    (x, y, z, angle) = pr.quaternion_diff(current_qxyzw, ref_qxyzw)
    error = (-1) * _lambda * angle * np.array([x,y,z])
    print('current | target | error | next')
    print('[{:+.4f}|{:+.4f}, {:+.4f}|{:+.4f}, {:+.4f}|{:+.4f}, {:+.4f}|{:+.4f}]'.format(current_pos[3], ref_pos[3], current_pos[4], ref_pos[4],current_pos[5], ref_pos[5],current_pos[6], ref_pos[6]))
    return error


def pbvs6(current_pos, ref_pos, _lambda_xyz=1.5, _lambda_orn=1.5):
    current_xyz = np.array(current_pos[0:3])
    ref_xyz = np.array(ref_pos[0:3]) + [0,0,0.005]
    residual_error = np.linalg.norm(ref_xyz - current_xyz)
    if residual_error<0.008:
        ref_xyz = np.array(ref_pos[0:3]) - [0, 0, 0.003]
    error_xyz = (ref_xyz - current_xyz)*_lambda_xyz
    # next_xyz = current_xyz + error_xyz
    current_qxyzw = change_qxyzw2qwxyz(current_pos[3:7])
    ref_qxyzw = change_qxyzw2qwxyz(ref_pos[3:7])
    # Compute the rotation in angle-axis format that rotates next_qxyzw into current_qxyzw.
    (x, y, z, angle) = pr.quaternion_diff(current_qxyzw, ref_qxyzw)
    error_orn = (-1) * _lambda_orn * angle * np.array([x, y, z])
    # print('current | target')
    # print('[xyz {:+.4f}] [{:+.4f}|{:+.4f}, {:+.4f}|{:+.4f}, {:+.4f}|{:+.4f}], q [{:+.4f}|{:+.4f}, {:+.4f}|{:+.4f}, {:+.4f}|{:+.4f}, {:+.4f}|{:+.4f}]'.format(residual_error, current_xyz[0], ref_xyz[0], current_xyz[1], ref_xyz[1], current_xyz[2], ref_xyz[2], current_pos[3], ref_pos[3], current_pos[4], ref_pos[4],current_pos[5], ref_pos[5],current_pos[6], ref_pos[6]))
    return np.hstack((error_xyz, error_orn))


def calculate_cumulants(current_pos, ref_pos, _lambda=1):
    current_xyz = np.array(current_pos[0:3])
    ref_xyz = np.array(ref_pos[0:3])
    current_qxyzw = change_qxyzw2qwxyz(current_pos[3:7])
    ref_qxyzw = change_qxyzw2qwxyz(ref_pos[3:7])
    (x, y, z, angle) = pr.quaternion_diff(current_qxyzw, ref_qxyzw)
    error_xyz = current_xyz - ref_xyz
    error_orn = angle * np.array([x, y, z])
    return (-1) * _lambda * np.hstack((error_xyz, error_orn))
