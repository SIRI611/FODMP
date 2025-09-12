import os
import math
import numpy as np
import random
from gym_env.utils.helpers import random_val_continuous, random_val_discreete
import transforms3d.quaternions as quat_transform


def generate_random_slot_pos(base_position, base_orn, xy_bounds=[0.05,0.05], yaw_bounds=[-math.pi, math.pi]):
    x, y, z = base_position
    roll, pitch, yaw = base_orn
    x += random.uniform(-xy_bounds[0], xy_bounds[1])
    y += random.uniform(-xy_bounds[0], xy_bounds[1])
    yaw = random.uniform(yaw_bounds[0], yaw_bounds[1])
    return [x, y, z], [math.radians(roll), math.radians(pitch), yaw]


def load_peg_prev(bullet_client, peg_slot_config, robot):
    peg_rgba_color = [0, 0.7, 0, 1]
    default_obj_mass = 0.01
    peg_multi_body_flags = bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | bullet_client.URDF_ENABLE_SLEEPING
    peg_file = os.environ['path-to-pegs'] + peg_slot_config['peg']['geometry'] + '.obj'
    peg_collision_shape = bullet_client.createCollisionShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=peg_file,
        meshScale=peg_slot_config['peg']['scale-factor']
    )
    peg_visual_shape = bullet_client.createVisualShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=peg_file,
        meshScale=peg_slot_config['peg']['scale-factor'],
        rgbaColor=peg_rgba_color
    )
    if peg_slot_config['peg']['spawn-in-hand']:
        finger_states = bullet_client.getLinkStates(robot.robot_id,robot.finger_indices)
        peg_position = [sum([finger_state[0][i] for finger_state in finger_states])/2 for i in range(3)]
        peg_position[2] -= 0.04
        peg_orientation_quat = [sum([finger_state[1][i] for finger_state in finger_states])/2 for i in range(4)]
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
        baseInertialFramePosition=[0, 0, 0],
        baseOrientation=peg_orientation_quat
    )

    if peg_slot_config['peg']['spawn-in-hand']:
        robot.open_gripper(0.005)
        robot.close_gripper()
    assert peg_id > -1, 'Multi body of peg could not be created'
    return peg_id

def load_peg_slot(bullet_client, peg_slot_config, robot, peg_rgba_color = None, slot_rgba_color = None, random_pos=True):
    """
    Loads a slot according to the slot configuration passed to this function. Configurable aspects of the slot are the
    wavefront OBJ file to load, the slot's position, the slot's orientation, and the factor to scale the slot's size by.
    The color of the bin can also be specified with the rgba_color parameter.

    Based on the load_bin function from bin_utils

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    peg_slot_config : dict
        Dictionary corresponding to the scene > peg-slot section of the config file for the peg and slot to load.
    robot : gym_env.components.picking_robot.PickingRobot
        Instance of the PickingRobot class.
    peg_rgba_color : list
        The color of the peg in RGBA format. Is green by default. List of length 4, floats in range [0,1]. 
    slot_rgba_color : list
        The color of the slot in RGBA format. Is light grey by default. List of length 4, floats in range [0,1]. 
    Returns
    -------
    tuple
        An tuple consisting of two integers. The first integer is the unique object id of the spawned peg and
        the second integer is the unnique object id of the spawned slot.
    """
    if peg_rgba_color is None:
        peg_rgba_color = [0, 0.7, 0, 1]
    if slot_rgba_color is None:
        slot_rgba_color = [0.6, 0.6, 0.6, 1]
    fixed_object_mass = 0
    default_obj_mass = 0.01
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
        finger_states = bullet_client.getLinkStates(robot.robot_id,robot.finger_indices)
        peg_position = [sum([finger_state[0][i] for finger_state in finger_states])/2 for i in range(3)]
        peg_position[2] -= 0.04
        peg_orientation_quat = [sum([finger_state[1][i] for finger_state in finger_states])/2 for i in range(4)]
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
        baseInertialFramePosition=[0, 0, 0],
        baseOrientation=peg_orientation_quat
    )

    if peg_slot_config['peg']['spawn-in-hand']:
        robot.open_gripper(0.005)
        robot.close_gripper()
    assert peg_id > -1, 'Multi body of peg could not be created'
    slot_flags = bullet_client.GEOM_FORCE_CONCAVE_TRIMESH
    slot_multi_body_flags = bullet_client.URDF_ENABLE_SLEEPING
    slot_visual_file = os.environ['path-to-slots'] + peg_slot_config['geometry'] + '.obj'
    slot_collision_file = os.environ['path-to-slots'] + 'collision/' + peg_slot_config['geometry'] + '.obj'
    if random_pos:
        slot_position, slot_orientation = generate_random_slot_pos(peg_slot_config['slot']['position'],
                                                                        peg_slot_config['slot']['orientation'])
    else:
        slot_position, slot_orientation = peg_slot_config['slot']['position'], \
                                          [math.radians(angle) for angle in peg_slot_config['slot']['orientation']]
    slot_orientation_quat = bullet_client.getQuaternionFromEuler(slot_orientation)
    slot_collision_shape = bullet_client.createCollisionShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=slot_collision_file,
        meshScale=peg_slot_config['scale-factor'],
        flags=slot_flags
    )
    slot_visual_shape = bullet_client.createVisualShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=slot_visual_file,
        meshScale=peg_slot_config['scale-factor'],
        rgbaColor=slot_rgba_color
    )
    slot_id = bullet_client.createMultiBody(
        baseMass=fixed_object_mass,
        baseCollisionShapeIndex=slot_collision_shape,
        baseVisualShapeIndex=slot_visual_shape,
        basePosition=slot_position,
        baseOrientation=slot_orientation_quat,
        flags=slot_multi_body_flags
    )
    assert slot_id > -1, 'Multi body for slot could not be created.'
    return peg_id, slot_id

def calc_rigid_body_coors(bullet_client, body_id, distance_to_origin, direction_vector=[0,0,1], debug=False):
    pos, orn = bullet_client.getBasePositionAndOrientation(body_id)
    from scipy.spatial.transform import Rotation as R
    r = R.from_quat(orn)
    new_direction = np.matmul(r.as_matrix(), direction_vector)
    end_point = np.array(pos) + distance_to_origin*new_direction
    if debug:
        bullet_client.addUserDebugLine(pos, end_point, lineColorRGB=[1,0,0], lineWidth=3.0)
    return end_point

# ============Codes for placement v4 ====================
def load_peg(bullet_client, peg_slot_config, robot, dimensions, peg_rgba_color=None):
    """
    Loads a peg according to the peg configuration passed to this function. Configurable aspects of the peg
    re the wavefront OBJ file to load, the peg's positions, the peg's orientations, and the factor to scale it's size by.
    The color of the peg can also be specified with the rgba_color parameter.

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    peg_slot_config : dict
        Dictionary corresponding to the scene > peg-slot section of the config file for the peg and slot to load.\
    dimensions:
        Dictionary containing the dimensions for each peg geometry type
    peg_rgba_color : list
        The color of the peg in RGBA format. Is green by default. List of length 4, floats in range [0,1].
    Returns
    -------
    tuple
        An tuple consisting of two integers. The first integer is the unique object id of the spawned peg,
        and the second is the height of the peg.
    """
    if peg_rgba_color is None:
        peg_rgba_color = [0, 0.7, 0, 1]
    peg_multi_body_flags = bullet_client.URDF_ENABLE_SLEEPING
    geometry = random_val_discreete(peg_slot_config['geometry'])
    scale_factor = peg_slot_config['scale-factor']
    try:
        scale_factor = random_val_continuous(scale_factor)
    except:
        scale_factor = [random_val_continuous(dim) for dim in scale_factor]
    if not isinstance(scale_factor, list):
        scale_factor = [scale_factor] * 3
    try:
        default_obj_mass = random_val_continuous(peg_slot_config['peg']['mass'])
    except:
        default_obj_mass = 0.01
    peg_file = os.environ['path-to-pegs'] + geometry + '.obj'
    peg_collision_shape = bullet_client.createCollisionShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=peg_file,
        meshScale=scale_factor
    )
    peg_visual_shape = bullet_client.createVisualShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=peg_file,
        meshScale=scale_factor,
        rgbaColor=peg_rgba_color
    )
    if peg_slot_config['peg']['spawn-in-hand']['enabled']:
        axis = random_val_discreete(peg_slot_config['peg']['spawn-in-hand']['axis'])
        angle = math.radians(random_val_continuous(peg_slot_config['peg']['spawn-in-hand']['angle']))
        offset = random_val_continuous(peg_slot_config['peg']['spawn-in-hand']['offset'])
        grasptarget_state = bullet_client.getLinkState(robot.robot_id, robot.grasptarget_index)
        target_orn_mat = np.array(bullet_client.getMatrixFromQuaternion(grasptarget_state[1])).reshape(3, 3)
        position_offset = np.array([0, 0, offset]) @ target_orn_mat.T
        peg_position = np.array(grasptarget_state[0]) + position_offset
        ax = [[0, 0, 0], [np.pi, 0, 0], [0, 0, np.pi / 2], [0, 0, -np.pi / 2], [np.pi / 2, 0, 0], [-np.pi / 2, 0, 0]][
            axis]
        pre_pre_rot = [1, 0, 0, 0]
        pre_rot = bullet_client.getQuaternionFromEuler(ax)
        peg_orn_quat = bullet_client.getQuaternionFromEuler((0, angle, 0))
        peg_orientation_quat = qmult(grasptarget_state[1], qmult(peg_orn_quat, qmult(pre_rot, pre_pre_rot)))
        finger_dist = dimensions[geometry][axis // 2] * scale_factor[axis // 2] / 2
        for joint_index in robot.finger_indices:
            bullet_client.resetJointState(robot.robot_id, joint_index, finger_dist)
        # robot.close_gripper()
    else:
        peg_position = [random_val_continuous(pos) for pos in peg_slot_config['peg']['position']]
        peg_orientation_quat = bullet_client.getQuaternionFromEuler(
            [math.radians(random_val_continuous(angle)) for angle in peg_slot_config['peg']['orientation']]
        )
    peg_id = bullet_client.createMultiBody(
        baseMass=default_obj_mass,
        baseCollisionShapeIndex=peg_collision_shape,
        baseVisualShapeIndex=peg_visual_shape,
        flags=peg_multi_body_flags,
        basePosition=peg_position,
        baseOrientation=peg_orientation_quat
    )
    if peg_slot_config['peg']['spawn-in-hand']['enabled']:
        robot.close_gripper()
    assert peg_id > -1, 'Multi body of peg could not be created'
    return peg_id


def load_slot(bullet_client, peg_slot_config, slot_rgba_color=None):
    """
    Loads a slot according to the slot configuration passed to this function. Configurable aspects of the slot
    are the wavefront OBJ file to load, the slot's position, the slot's orientation, and the factor to scale it's size by.
    The color of the slot can also be specified with the rgba_color parameter.

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    peg_slot_config : dict
        Dictionary corresponding to the scene > peg-slot section of the config file for the peg and slot to load.
    slot_rgba_color : list
        The color of the slot in RGBA format. Is light grey by default. List of length 4, floats in range [0,1].
    Returns
    -------
    tuple
        The first element is an integer corresponding to the uniqur object id of the spawned slot. The second element is a
        dict that contains the same fields as the input config, but with the elements that were used for the slot, this is
        useful for spawning the peg if one of their common elements was to be randomly determined.
    """
    if slot_rgba_color is None:
        slot_rgba_color = [0.7, 0.7, 0.7, 1]
    fixed_object_mass = 0
    geometry = random_val_discreete(peg_slot_config['geometry'])
    scale_factor = peg_slot_config['scale-factor']
    try:
        scale_factor = random_val_continuous(scale_factor)
    except:
        scale_factor = [random_val_continuous(dim) for dim in scale_factor]
    if not isinstance(scale_factor, list):
        scale_factor = [scale_factor] * 3
    slot_flags = bullet_client.GEOM_FORCE_CONCAVE_TRIMESH
    slot_multi_body_flags = bullet_client.URDF_ENABLE_SLEEPING
    slot_visual_file = os.environ['path-to-slots'] + geometry + '.obj'
    slot_collision_file = os.environ['path-to-slots'] + 'collision/' + geometry + '.obj'
    base_visual_file = os.environ['path-to-slots'] + "base_single.obj"
    base_collision_file = os.environ['path-to-slots'] + "collision/base_single.obj"
    slot_position = [random_val_continuous(pos) for pos in peg_slot_config['slot']['position']]
    slot_orientation_euler = [random_val_continuous(angle) for angle in peg_slot_config['slot']['orientation']]
    slot_orientation_quat = bullet_client.getQuaternionFromEuler([math.radians(i) for i in slot_orientation_euler])
    slot_collision_shape = bullet_client.createCollisionShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=slot_collision_file,
        meshScale=scale_factor,
        flags=slot_flags
    )
    slot_visual_shape = bullet_client.createVisualShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=slot_visual_file,
        meshScale=scale_factor,
        rgbaColor=slot_rgba_color
    )
    base_collision_shape = bullet_client.createCollisionShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=base_collision_file,
        meshScale=scale_factor,
        flags=slot_flags
    )
    base_visual_shape = bullet_client.createVisualShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=base_visual_file,
        meshScale=scale_factor,
        rgbaColor=slot_rgba_color
    )
    slot_id = bullet_client.createMultiBody(
        baseMass=fixed_object_mass,
        baseCollisionShapeIndex=base_collision_shape,
        baseVisualShapeIndex=base_visual_shape,
        basePosition=slot_position,
        baseOrientation=slot_orientation_quat,
        flags=slot_multi_body_flags,
        linkMasses=[fixed_object_mass],
        linkCollisionShapeIndices=[slot_collision_shape],
        linkVisualShapeIndices=[slot_visual_shape],
        linkPositions=[[0, 0, 0]],
        linkOrientations=[[0, 0, 0, 1]],
        linkInertialFramePositions=[[0, 0, 0]],
        linkInertialFrameOrientations=[[0, 0, 0, 1]],
        linkParentIndices=[0],
        linkJointTypes=[bullet_client.JOINT_FIXED],
        linkJointAxis=[[0, 0, 1]]
    )
    assert slot_id > -1, 'Multi body for slot could not be created.'
    info = {
        'geometry': geometry,
        'scale-factor': scale_factor,
        'slot': {
            'position': slot_position,
            'orientation': slot_orientation_euler,
        }
    }
    return slot_id, info


def load_multi_peg_slot(bullet_client, peg_slot_config, robot, dimensions, peg_rgba_color=None, slot_rgba_color=None):
    """
    Loads a slot and peg, which are composed of several sub pegs and slots, according to the peg and slot configuration
    passed to this function. Configurable aspects of the slotand peg are the wavefront OBJ files to load, their positions,
    their orientations, and the factor to scale their size by.
    The color of the peg and slot can also be specified with the rgba_color parameter.

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    peg_slot_config : dict
        Dictionary corresponding to the scene > peg-slot section of the config file for the peg and slot to load.
    robot : gym_env.components.picking_robot.PickingRobot
        Instance of the PickingRobot class.
    dimensions:
        Dictionary containing the dimensions for each peg geometry type
    peg_rgba_color : list
        The color of the peg in RGBA format. Is green by default. List of length 4, floats in range [0,1].
    slot_rgba_color : list
        The color of the slot in RGBA format. Is light grey by default. List of length 4, floats in range [0,1].
    Returns
    -------
    tuple
        An tuple consisting of four integers. The first integer is the unique object id of the spawned peg,
        the second integer is the unnique object id of the spawned slot, the third is the height of the peg
        and the fourth is the number of sub pegs/slots that were used.
    """
    peg_config = {
        'linkMasses': [],
        'linkCollisionShapeIndices': [],
        'linkVisualShapeIndices': [],
        'linkPositions': [],
    }
    slot_config = {
        'linkMasses': [],
        'linkCollisionShapeIndices': [],
        'linkVisualShapeIndices': [],
        'linkPositions': [],
    }
    common_config = {
        'linkOrientations': [],
        'linkInertialFramePositions': [],
        'linkInertialFrameOrientations': [],
        'linkParentIndices': [],
        'linkJointTypes': [],
        'linkJointAxis': [],
    }
    fixed_object_mass = 0
    if peg_rgba_color is None:
        peg_rgba_color = [0, 0.7, 0, 1]
    if slot_rgba_color is None:
        slot_rgba_color = [0.7, 0.7, 0.7, 0.7]
    slot_flags = bullet_client.GEOM_FORCE_CONCAVE_TRIMESH
    multi_body_flags = bullet_client.URDF_ENABLE_SLEEPING
    geometry = [random_val_discreete(geometry) for geometry in peg_slot_config['geometry']]
    scale_factor = peg_slot_config['scale-factor']
    for i in range(len(scale_factor)):
        try:
            scale_factor[i] = [random_val_continuous(scale_factor[i])] * 3
        except:
            scale_factor[i] = [random_val_continuous(dim) for dim in scale_factor[i]]
    angle = [random_val_continuous(angle) for angle in peg_slot_config['angle']]
    assert len(geometry) == len(scale_factor) == len(
        angle), 'For a multi_peg the geometry, scale-factor and angle list must have the same length.'
    offset = [random_val_continuous(offset) for offset in peg_slot_config['offset']]
    assert len(offset) + 1 == len(geometry)
    try:
        default_obj_mass = random_val_continuous(peg_slot_config['peg']['mass'])
    except:
        default_obj_mass = 0.01
    peg_width = sum(offset) + 0.02
    offset.insert(0, 0.01)
    position = -peg_width / 2
    peg_heights = [(dimensions[geometry[i]][2]) * scale_factor[i][2] for i in range(len(geometry))]
    slot_depth_need = [
        dimensions['slot'][2] * scale_factor[i][2] if dimensions[geometry[i]][3] else peg_heights[i] / 2 +
                                                                                      dimensions['slot'][2] *
                                                                                      scale_factor[i][2] for i in
        range(len(peg_heights))]
    clearance = max(slot_depth_need)
    peg_dims = [0.02, peg_width, 0.02 + max(peg_heights)]
    for sub_peg_num in range(len(geometry)):
        common_config['linkOrientations'].append(
            bullet_client.getQuaternionFromEuler([0, 0, math.radians(angle[sub_peg_num])]))
        common_config['linkInertialFramePositions'].append([0, 0, 0])
        common_config['linkInertialFrameOrientations'].append([0, 0, 0, 1])
        common_config['linkParentIndices'].append(0)
        common_config['linkJointTypes'].append(bullet_client.JOINT_FIXED)
        common_config['linkJointAxis'].append([0, 0, 1])
        position += offset[sub_peg_num]
        sub_peg_file = os.environ['path-to-pegs'] + geometry[sub_peg_num] + '.obj'
        sub_peg_collision_shape = bullet_client.createCollisionShape(
            shapeType=bullet_client.GEOM_MESH,
            fileName=sub_peg_file,
            meshScale=scale_factor[sub_peg_num]
        )
        sub_peg_visual_shape = bullet_client.createVisualShape(
            shapeType=bullet_client.GEOM_MESH,
            fileName=sub_peg_file,
            meshScale=scale_factor[sub_peg_num],
            rgbaColor=peg_rgba_color
        )
        peg_config['linkMasses'].append(default_obj_mass)
        peg_config['linkCollisionShapeIndices'].append(sub_peg_collision_shape)
        peg_config['linkVisualShapeIndices'].append(sub_peg_visual_shape)
        peg_config['linkPositions'].append([position, 0, -peg_heights[sub_peg_num] / 2 - 0.01])
        sub_slot_visual_file = os.environ['path-to-slots'] + geometry[sub_peg_num] + '.obj'
        sub_slot_collision_file = os.environ['path-to-slots'] + 'collision/' + geometry[sub_peg_num] + '.obj'
        sub_slot_collision_shape = bullet_client.createCollisionShape(
            shapeType=bullet_client.GEOM_MESH,
            fileName=sub_slot_collision_file,
            meshScale=scale_factor[sub_peg_num],
            flags=slot_flags
        )
        sub_slot_visual_shape = bullet_client.createVisualShape(
            shapeType=bullet_client.GEOM_MESH,
            fileName=sub_slot_visual_file,
            meshScale=scale_factor[sub_peg_num],
            rgbaColor=slot_rgba_color
        )
        slot_config['linkMasses'].append(fixed_object_mass)
        slot_config['linkCollisionShapeIndices'].append(sub_slot_collision_shape)
        slot_config['linkVisualShapeIndices'].append(sub_slot_visual_shape)
        slot_config['linkPositions'].append([position, 0, clearance - slot_depth_need[sub_peg_num]])
    if peg_slot_config['peg']['spawn-in-hand']['enabled']:
        angle = math.radians(random_val_continuous(peg_slot_config['peg']['spawn-in-hand']['angle']))
        offset = random_val_continuous(peg_slot_config['peg']['spawn-in-hand']['offset'])
        grasptarget_state = bullet_client.getLinkState(robot.robot_id, robot.grasptarget_index)
        target_orn_mat = np.array(bullet_client.getMatrixFromQuaternion(grasptarget_state[1])).reshape(3, 3)
        position_offset = np.array([0, 0, offset]) @ target_orn_mat.T
        peg_position = np.array(grasptarget_state[0]) + position_offset
        pre_rot = bullet_client.getQuaternionFromEuler([np.pi, 0, 0])
        peg_orn_quat = bullet_client.getQuaternionFromEuler((0, angle, 0))
        peg_orientation_quat = qmult(grasptarget_state[1], qmult(peg_orn_quat, pre_rot))
        finger_dist = 0.01
        for joint_index in robot.finger_indices:
            bullet_client.resetJointState(robot.robot_id, joint_index, finger_dist)
    else:
        peg_position = [random_val_continuous(pos) for pos in peg_slot_config['peg']['position']]
        peg_orientation_quat = bullet_client.getQuaternionFromEuler(
            [math.radians(random_val_continuous(angle)) for angle in peg_slot_config['peg']['orientation']]
        )
    peg_collision_shape = bullet_client.createCollisionShape(
        shapeType=bullet_client.GEOM_BOX,
        halfExtents=[peg_width / 2, 0.01, 0.01]
    )
    peg_visual_shape = bullet_client.createVisualShape(
        shapeType=bullet_client.GEOM_BOX,
        halfExtents=[peg_width / 2, 0.01, 0.01],
        rgbaColor=peg_rgba_color
    )
    peg_id = bullet_client.createMultiBody(
        baseMass=default_obj_mass,
        baseCollisionShapeIndex=peg_collision_shape,
        baseVisualShapeIndex=peg_visual_shape,
        flags=multi_body_flags,
        basePosition=peg_position,
        baseOrientation=peg_orientation_quat,
        **peg_config,
        **common_config
    )
    if peg_slot_config['peg']['spawn-in-hand']['enabled']:
        robot.close_gripper()
    slot_position = [random_val_continuous(pos) for pos in peg_slot_config['slot']['position']]
    slot_orientation_quat = bullet_client.getQuaternionFromEuler(
        [math.radians(random_val_continuous(angle)) for angle in peg_slot_config['slot']['orientation']]
    )
    slot_id = bullet_client.createMultiBody(
        baseMass=fixed_object_mass,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=-1,
        basePosition=slot_position,
        baseOrientation=slot_orientation_quat,
        flags=multi_body_flags,
        **slot_config,
        **common_config
    )
    return peg_id, slot_id, max(peg_heights), len(geometry)


def load_base_slot(bullet_client, peg_slot_config, slot_rgba_color = None):
    """
    Loads a base contatinign 4 slots according to the slot configuration passed to this function. Configurable aspects of the slot
    are the wavefront OBJ files to load, the base's position, the base's orientation, as well as the type scale and rotation of the
    individual slots.
    The color of the slot can also be specified with the rgba_color parameter.

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    peg_slot_config : dict
        Dictionary corresponding to the scene > peg-slot section of the config file for the peg and slot to load.
    slot_rgba_color : list
        The color of the slot in RGBA format. Is light grey by default. List of length 4, floats in range [0,1].
    Returns
    -------
    tuple
        The first element is an integer corresponding to the uniqur object id of the spawned slot. The second element is a
        list of dicts for each slot that contains the same fields as the input config, but with the elements that were used for that slot,
        this is useful for spawning the peg if one of their common elements was to be randomly determined.
    """
    slots_config = {
        'linkMasses':[],
        'linkCollisionShapeIndices':[],
        'linkVisualShapeIndices':[],
        'linkPositions':[],
        'linkOrientations':[],
        'linkInertialFramePositions':[],
        'linkInertialFrameOrientations':[],
        'linkParentIndices':[],
        'linkJointTypes':[],
        'linkJointAxis':[],
    }
    fixed_object_mass = 0
    if slot_rgba_color is None:
        slot_rgba_color = [0.5, 0.5, 0.5, 1]
    slot_flags = bullet_client.GEOM_FORCE_CONCAVE_TRIMESH
    multi_body_flags = bullet_client.URDF_ENABLE_SLEEPING
    geometry = [random_val_discreete(geometry) for geometry in peg_slot_config['geometry']]
    scale_factor = peg_slot_config['scale-factor']
    for i in range(len(scale_factor)):
        try:
            scale_factor[i] = [random_val_continuous(scale_factor[i])]*3
        except:
            scale_factor[i] = [random_val_continuous(dim) for dim in scale_factor[i]]
    try:
        overall_scale_factor = [random_val_discreete(geometry) for geometry in peg_slot_config['overall-scale-factor']]
    except:
        overall_scale_factor = [1, 1, 1]
    try:
        angle = [random_val_continuous(angle) for angle in peg_slot_config['angle']]
    except:
        angle = [0,0,0,0]
    assert len(geometry) == len(scale_factor) == len(angle) == 4, 'For the base with slots the geometry,' \
        'scale-factor and angle (if used) list must be of length 4.'
    assert scale_factor[0][-1] == scale_factor[1][-1] == scale_factor[2][-1] == scale_factor[3][-1], \
        'All slots must have the same vertical scale factor'
    scale_factor = [
        [i[0] * overall_scale_factor[0], i[1] * overall_scale_factor[1], i[2] * overall_scale_factor[2]]
        for i in scale_factor
    ]
    positions = [
        [0.017 * overall_scale_factor[0], 0.017 * overall_scale_factor[1], 0],
        [0.017 * overall_scale_factor[0], -0.017 * overall_scale_factor[1], 0],
        [-0.017 * overall_scale_factor[0], -0.017 * overall_scale_factor[1], 0],
        [-0.017 * overall_scale_factor[0], 0.017 * overall_scale_factor[1], 0]
    ]
    for slot_num in range(4):
        slot_visual_file = os.environ['path-to-slots'] + geometry[slot_num] + '.obj'
        slot_collision_file = os.environ['path-to-slots'] + 'collision/' + geometry[slot_num] + '.obj'
        slot_collision_shape = bullet_client.createCollisionShape(
            shapeType=bullet_client.GEOM_MESH,
            fileName=slot_collision_file,
            meshScale=scale_factor[slot_num],
            flags=slot_flags
        )
        slot_visual_shape = bullet_client.createVisualShape(
            shapeType=bullet_client.GEOM_MESH,
            fileName=slot_visual_file,
            meshScale=scale_factor[slot_num],
            rgbaColor=slot_rgba_color
        )
        slots_config['linkMasses'].append(fixed_object_mass)
        slots_config['linkCollisionShapeIndices'].append(slot_collision_shape)
        slots_config['linkVisualShapeIndices'].append(slot_visual_shape)
        slots_config['linkPositions'].append(positions[slot_num])
        slots_config['linkOrientations'].append(bullet_client.getQuaternionFromEuler([0,0,math.radians(angle[slot_num])]))
        slots_config['linkInertialFramePositions'].append([0,0,0])
        slots_config['linkInertialFrameOrientations'].append([0,0,0,1])
        slots_config['linkParentIndices'].append(0)
        slots_config['linkJointTypes'].append(bullet_client.JOINT_FIXED)
        slots_config['linkJointAxis'].append([0,0,1])
    slot_position = peg_slot_config['slot']['position']  # [random_val_continuous(pos) for pos in peg_slot_config['slot']['position']]
    slot_orientation = [math.radians(random_val_continuous(angle)) for angle in peg_slot_config['slot']['orientation']]
    slot_orientation_quat = bullet_client.getQuaternionFromEuler(slot_orientation)
    base_file = os.environ['path-to-slots'] + 'base.obj'
    base_collision_shape = bullet_client.createCollisionShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=base_file,
        meshScale=overall_scale_factor,
        flags=slot_flags
    )
    base_visual_shape = bullet_client.createVisualShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=base_file,
        meshScale=overall_scale_factor,
        rgbaColor=slot_rgba_color
    )
    slot_id = bullet_client.createMultiBody(
        baseMass=fixed_object_mass,
        baseCollisionShapeIndex=base_collision_shape,
        baseVisualShapeIndex=base_visual_shape,
        basePosition=slot_position,
        baseOrientation=slot_orientation_quat,
        flags=multi_body_flags,
        **slots_config,
    )
    info = [{
        'geometry': geometry[i],
        'scale-factor': scale_factor[i],
        'slot': {
            'position': [i+j for i,j in zip(slot_position,rotate_vector(positions[i],slot_orientation_quat))],
            'orientation': [math.degrees(j) for j in slot_orientation[:-1] + [slot_orientation[-1] + angle[i]]]
        }} for i in range(4)]
    return slot_id, info


def check_peg_placement(bullet_client, peg_id, slot_id, peg_dimensions, geometry, scale_factor, angle_tol, dist_tol,
                        slot_index=0):
    """
    Checks to see if a peg has been correctly placed in a slot, and returns relevant info.

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    peg_id : dict
        Dictionary corresponding to the scene > peg-slot section of the config file for the peg and slot to load.
    peg_dimensions:
        Dictionary containing the dimensions for each peg geometry type
    geometry : string
        The color of the peg in RGBA format. Is green by default. List of length 4, floats in range [0,1].
    scale_factor : list
        The scale of the peg as an integer for each axis [x,y,z].
    angle_tol : integer
        The angle tolerance in degrees for peg placement.
    dist_tol : integer
        The distance tolerance for peg placement.
    slot_index : index
        The index of the desired slot in the slot links, relevant for a base configuration which contains multiple slots.
        For single slot bases it can be excluded since it defaults to 0.
    Returns
    -------
    dict
        A dictionary containing info on wether the specified peg has been placed into the specified slot.
    """
    info = {'successful-placement': False, 'failed-placement': False, 'positioned-peg': False}
    peg_pos, peg_orn = bullet_client.getBasePositionAndOrientation(peg_id)
    slot_pos, slot_orn, _, _, _, _ = bullet_client.getLinkState(slot_id, slot_index)
    dist = np.sqrt(np.power(np.array(peg_pos[:-1]) - np.array(slot_pos[:-1]), 2).sum())
    height = peg_pos[2] - slot_pos[2] - peg_dimensions[geometry][2] * scale_factor[2] / 2
    if dist < 0.04:  # peg is close to slot
        if height < peg_dimensions['slot'][2] * scale_factor[2] + 0.01:  # peg is above the slot
            slot_orn = (-slot_orn[0], -slot_orn[1], -slot_orn[2], slot_orn[3])
            # calculate peg orientation vs slot orientation
            sym_x, sym_y, sym_z = peg_dimensions[geometry][3:6]
            oriented = False
            for y_rot in range(sym_y):
                new_orn = qmult(peg_orn, bullet_client.getQuaternionFromEuler((0, y_rot * 2 * np.pi / sym_y, 0)))
                diff_orn = qmult(new_orn, slot_orn)
                diff_x, diff_y, diff_z = [np.degrees(i) for i in bullet_client.getEulerFromQuaternion(diff_orn)]
                if abs(diff_y) < angle_tol:
                    if (abs(diff_x) % (360 / sym_x) < angle_tol or 360 / sym_x - abs(diff_x) % (
                            360 / sym_x) < angle_tol) \
                            and (abs(diff_z) % (360 / sym_z) < angle_tol or 360 / sym_z - abs(diff_z) % (
                            360 / sym_z) < angle_tol):
                        oriented = True
                        break
            if height < peg_dimensions['slot'][2] * scale_factor[2]:  # peg is in the slot
                # check distance is less than tolerance and that peg is oriented
                if dist < dist_tol and oriented:
                    if height < peg_dimensions['slot'][3] * scale_factor[2]:  # peg has been inserted
                        info['successful-placement'] = True
                else:
                    info['failed-placement'] = True
            elif dist < dist_tol and oriented:
                info['positioned-peg'] = True
    return info


def qmult(q1, q2):
    """
    The transforms3d.quaternions uses a quaternion format of (w,x,y,z) and pybullet uses (x,y,z,w)
    and this function performs quaternion multiplication from transforms3d.quaternions taking and
    returning pybullet quaternions.

    Parameters
    ----------
    q1: array-like
        Quaternion of the format (x,y,z,w)
    q2: array-like
        Quaternion of the format (x,y,z,w)
    Returns
    -------
    tuple
        An tuple consisting of 4 floats. It represents a quaternion of the form (x,y,z,w)
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w, x, y, z = quat_transform.qmult((w1, x1, y1, z1), (w2, x2, y2, z2))
    return (x, y, z, w)


def rotate_vector(v, q):
    """
    The transforms3d.quaternions uses a quaternion format of (w,x,y,z) and pybullet uses (x,y,z,w)
    and this function performs a vector rotation by a quaternion from transforms3d.quaternions taking a
    pybullet quaternion.

    Parameters
    ----------
    q: array-like
        Quaternion of the format (x,y,z,w)
    Returns
    -------
    tuple
        An tuple consisting of 3 floats. It represents a vector
    """
    x, y, z, w = q
    return quat_transform.rotate_vector(v, (w, x, y, z))
