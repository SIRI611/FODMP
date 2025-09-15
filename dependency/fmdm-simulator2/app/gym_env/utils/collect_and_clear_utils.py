import os
import math
import numpy as np
import random
from gym_env.utils.helpers import random_val_continuous, random_val_discreete
import transforms3d.quaternions as quat_transform
from gym_env.utils.scene_utils import new_scene_obj_list

BASE=[ 1,-0.1,0.045 ]



def generate_random_obj_pos(base_position, num_pos=4, xy_bounds=[0.5,0.5]):
    x, y, z = base_position
    dx = [-0.5, -0.5, -0.6, -0.6]  #np.random.uniform(-xy_bounds[0], xy_bounds[1], num_pos)
    dy = [-0.1, -0.4, 0.1, -0.3]# np.random.uniform(-xy_bounds[0], xy_bounds[1], num_pos)
    pos_array = []
    pos_array_rand = []
    for i in range(num_pos):
        pos_array_rand.append([x+dx[i]+random.uniform(-0.05, 0.05), y+dy[i]+random.uniform(-0.05, 0.05), z])
        pos_array.append([x+dx[i], y+dy[i], z])
    return np.array(pos_array_rand), np.array(pos_array)

def reset_target_disruptive_objects(bullet_client, target_obj_infos, disrpt_obj_infos):
    random_obj_pos, original_pos = generate_random_obj_pos(BASE)
    pos_indx = [i for i in range(4)]
    random.shuffle(pos_indx)
    i_count = 0
    for obj in target_obj_infos:
        reset_object(bullet_client, obj['obj_id'], original_pos[pos_indx[i_count]])
        i_count +=1
    for disrpt_obj in disrpt_obj_infos:
        reset_object(bullet_client, disrpt_obj['obj_id'], original_pos[pos_indx[i_count]])
        i_count += 1

def reset_object(bullet_client, object_id, original_pos):
    pos, orn = bullet_client.getBasePositionAndOrientation(object_id)

    pos = [original_pos[0]+random.uniform(-0.05, 0.05), original_pos[1]+random.uniform(-0.05, 0.05), original_pos[2]]
    bullet_client.resetBasePositionAndOrientation(object_id, pos, orn)

def remove_objects(bullet_client, object_list):
    for obj in object_list:
        print(bullet_client.removeBody(obj['obj_id']))

def load_target_and_disruptive_objects(bullet_client, target_object_configs, disruptive_object_configs, target_color=[1,1,1,1], disruptive_color=[0.43921569, 0.38823529, 0.25098039, 1]):
    target_obj_infos = []
    disruptive_obj_infos = []
    # generate 4 random pos
    random_obj_pos, original_pos = generate_random_obj_pos(BASE)
    pos_indx = [i for i in range(4)]
    random.shuffle(pos_indx)
    # num_target_objs=random.randint(1,4)
    # num_disruptive_objects = 4-num_target_objs
    num_target_objs = target_object_configs['num-of-objects'][0]
    num_disruptive_objects = 4- num_target_objs
    print('num target obj {}, num disruptive obj {}'.format(num_target_objs, num_disruptive_objects))
    target_geometry = target_object_configs['geometry'][0]
    print(target_object_configs)
    print(disruptive_object_configs)
    for i in range(num_target_objs):
        target_obj_info = load_one_object(bullet_client, target_geometry, random_obj_pos[pos_indx[i]],
                                          original_pos[pos_indx[i]], target_object_configs['orientation'][0],
                                          target_object_configs['dimensions'][0],
                                          target_object_configs['scale-factor'][0],
                                          target_object_configs['collision-scale'][0],
                                          target_object_configs['object-mass'][0],
                                          color=target_color)
        target_obj_infos.append(target_obj_info)
    disrpt_obj_geometry = disruptive_object_configs['geometry'][0]
    for i in range(num_disruptive_objects):
        disruptive_obj_info = load_one_object(bullet_client, disrpt_obj_geometry,
                                              random_obj_pos[pos_indx[num_target_objs + i]],
                                              original_pos[pos_indx[num_target_objs + i]],
                                              disruptive_object_configs['orientation'][0],
                                              disruptive_object_configs['dimensions'][0],
                                              disruptive_object_configs['scale-factor'][0],
                                              disruptive_object_configs['collision-scale'][0],
                                              disruptive_object_configs['object-mass'][0], color=disruptive_color)
        disruptive_obj_infos.append(disruptive_obj_info)
    print(target_obj_infos)
    return target_obj_infos, disruptive_obj_infos


def load_one_object(bullet_client, geometry, random_obj_pos, pos_original, obj_orientation, dimension, obj_scale, collision_scale, obj_mass, color=None):
    obj_info = {}
    obj_file = os.environ['path-to-new-scenes'] + geometry + '/' + new_scene_obj_list[geometry]
    collision_file = os.environ['path-to-assets'] + 'pegs' + '/cylinder.obj'

    obj_id = load_single_object(bullet_client, obj_file, obj_scale,
                                obj_mass, random_obj_pos,
                                obj_orientation,
                                rgba_color = color,
                                collision_scale=collision_scale,
                                collision_obj_file=collision_file)
    obj_info['positions'] = pos_original
    obj_info['obj_id'] = obj_id
    obj_info['dimension'] = dimension
    obj_info['geomemtry'] = geometry
    obj_info['status'] = 0
    return obj_info

def load_objects_to_collect(bullet_client, object_configs):
    obj_infos = []
    for i in range(len(object_configs['geometry'])):
        geometry = object_configs['geometry'][i]
        obj_file = os.environ['path-to-new-scenes'] + geometry + '/' + new_scene_obj_list[geometry]
        collision_file = os.environ['path-to-assets'] + 'pegs' + '/cylinder.obj'
        pos_array_rand, pos_array = generate_random_obj_pos(object_configs['position'][i], object_configs['num-of-objects'][i])
        for j in range(object_configs['num-of-objects'][i]):
            # random obj position
            obj_info = {}
            random_obj_pos = pos_array[j]
            obj_id = load_single_object(bullet_client, obj_file, object_configs['scale-factor'][i],
                                        object_configs['object-mass'][i], random_obj_pos,
                                        object_configs['orientation'][i], collision_scale=object_configs['collision-scale'][i], collision_obj_file=collision_file)
            obj_info['positions']= pos_array[j]
            obj_info['obj_id'] = obj_id
            obj_info['dimension'] = object_configs['dimensions'][i]
            obj_info['geomemtry'] = geometry
            obj_infos.append(obj_info)
    return obj_infos


def load_container(bullet_client, container_configs, container_idx=0):
    fixed_object_mass = 0
    slot_rgba_color = [0.32941176, 0.47843137, 0.33333333, 1]
    # geometry = random_val_discreete(container_configs['geometry'])
    geometry = container_configs['geometry'][container_idx]  # test trash Can
    scale_factor = container_configs['scale-factor'][container_idx]
    container_flags = bullet_client.GEOM_FORCE_CONCAVE_TRIMESH
    container_multi_body_flags = bullet_client.URDF_ENABLE_SLEEPING
    container_visual_file = os.environ['path-to-new-scenes'] + geometry + '/' + new_scene_obj_list[geometry]
    container_collision_file = os.environ['path-to-new-scenes'] + geometry + '/' + new_scene_obj_list[geometry]
    base_visual_file = os.environ['path-to-new-scenes'] + geometry + '/' + new_scene_obj_list[geometry]
    base_collision_file = os.environ['path-to-assets'] + 'bins' + '/bin0.obj'
    container_position =container_configs['position'][container_idx]
    container_orientation_euler = container_configs['orientation'][container_idx]
    container_orientation_quat = bullet_client.getQuaternionFromEuler([math.radians(i) for i in container_orientation_euler])
    container_collision_shape = bullet_client.createCollisionShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=container_collision_file,
        meshScale=scale_factor,
        flags=container_flags
    )
    container_visual_shape = bullet_client.createVisualShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=container_visual_file,
        meshScale=scale_factor,
        rgbaColor=slot_rgba_color
    )
    base_collision_shape = bullet_client.createCollisionShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=base_collision_file,
        meshScale=[1.25,2.19,5],
        flags=container_flags
    )
    base_visual_shape = bullet_client.createVisualShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=base_visual_file,
        meshScale=scale_factor,
        rgbaColor=slot_rgba_color
    )
    container_id = bullet_client.createMultiBody(
        baseMass=fixed_object_mass,
        baseCollisionShapeIndex=base_collision_shape,
        baseVisualShapeIndex=base_visual_shape,
        basePosition=container_position,
        baseOrientation=container_orientation_quat,
        flags=container_multi_body_flags,
        linkMasses=[fixed_object_mass],
        linkCollisionShapeIndices=[container_collision_shape],
        linkVisualShapeIndices=[container_visual_shape],
        linkPositions=[[0, 0, 0]],
        linkOrientations=[[0, 0, 0, 1]],
        linkInertialFramePositions=[[0, 0, 0]],
        linkInertialFrameOrientations=[[0, 0, 0, 1]],
        linkParentIndices=[0],
        linkJointTypes=[bullet_client.JOINT_FIXED],
        linkJointAxis=[[0, 0, 1]]
    )
    assert container_id > -1, 'Multi body for slot could not be created.'
    info = {
        'geometry': geometry,
        'scale-factor': scale_factor,
        'container': {
            'position': container_position,
            'orientation': container_orientation_euler,
        }
    }
    return container_id, info

def load_plate(bullet_client, container_configs, container_idx=0):
    fixed_object_mass = 0
    slot_rgba_color = [1, 1, 1, 0]
    # geometry = random_val_discreete(container_configs['geometry'])
    geometry = container_configs['geometry'][container_idx]  # test trash Can
    scale_factor = container_configs['scale-factor'][container_idx]
    container_flags = bullet_client.GEOM_FORCE_CONCAVE_TRIMESH
    container_multi_body_flags = bullet_client.URDF_ENABLE_SLEEPING
    container_visual_file = os.environ['path-to-new-scenes'] + geometry + '/' + new_scene_obj_list[geometry]
    container_collision_file = os.environ['path-to-new-scenes'] + geometry + '/' + new_scene_obj_list[geometry]
    base_visual_file = os.environ['path-to-new-scenes'] + geometry + '/' + new_scene_obj_list[geometry]
    base_collision_file = os.environ['path-to-new-scenes'] + geometry + '/' + new_scene_obj_list[geometry]
    container_position =container_configs['position'][container_idx]
    container_orientation_euler = container_configs['orientation'][container_idx]
    container_orientation_quat = bullet_client.getQuaternionFromEuler([math.radians(i) for i in container_orientation_euler])
    container_collision_shape = bullet_client.createCollisionShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=container_collision_file,
        meshScale=scale_factor,
        flags=container_flags
    )
    container_visual_shape = bullet_client.createVisualShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=container_visual_file,
        meshScale=scale_factor,
        rgbaColor=slot_rgba_color
    )
    base_collision_shape = bullet_client.createCollisionShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=base_collision_file,
        meshScale=[1.25,2.19,5],
        flags=container_flags
    )
    base_visual_shape = bullet_client.createVisualShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=base_visual_file,
        meshScale=scale_factor,
        rgbaColor=slot_rgba_color
    )
    container_id = bullet_client.createMultiBody(
        baseMass=fixed_object_mass,
        baseCollisionShapeIndex=base_collision_shape,
        baseVisualShapeIndex=base_visual_shape,
        basePosition=container_position,
        baseOrientation=container_orientation_quat,
        flags=container_multi_body_flags,
        linkMasses=[fixed_object_mass],
        linkCollisionShapeIndices=[container_collision_shape],
        linkVisualShapeIndices=[container_visual_shape],
        linkPositions=[[0, 0, 0]],
        linkOrientations=[[0, 0, 0, 1]],
        linkInertialFramePositions=[[0, 0, 0]],
        linkInertialFrameOrientations=[[0, 0, 0, 1]],
        linkParentIndices=[0],
        linkJointTypes=[bullet_client.JOINT_FIXED],
        linkJointAxis=[[0, 0, 1]]
    )
    assert container_id > -1, 'Multi body for slot could not be created.'
    info = {
        'geometry': geometry,
        'scale-factor': scale_factor,
        'container': {
            'position': container_position,
            'orientation': container_orientation_euler,
        }
    }
    return container_id, info

def load_background_objects(bullet_client, object_configs):
    for i in range(len(object_configs['geometry'])):
        geometry = object_configs['geometry'][i]
        obj_file = os.environ['path-to-new-scenes'] + geometry + '/' + new_scene_obj_list[geometry]
        obj_id = load_single_object(bullet_client, obj_file, object_configs['scale-factor'][i],
                                    object_configs['object-mass'][i], object_configs["position"][i],
                                    object_configs['orientation'][i],
                                    object_configs['colors'][i])

    return None

def load_single_object(bullet_client, obj_file, scale_factor, mass, position, orientation, rgba_color=None, collision_scale=None, collision_obj_file=None):
    # scale_factor = [0.2, 0.2, 0.2]
    print(obj_file)
    obj_multi_body_flags = bullet_client.URDF_ENABLE_SLEEPING
    if collision_scale == None:
        obj_collision_shape = bullet_client.createCollisionShape(
            shapeType=bullet_client.GEOM_MESH,
            fileName=obj_file,
            meshScale=scale_factor
        )
    else:
        obj_collision_shape = bullet_client.createCollisionShape(
            shapeType=bullet_client.GEOM_MESH,
            fileName=obj_file,  #collision_obj_file,
            meshScale=scale_factor #collision_scale
        )
    if rgba_color == None or rgba_color == [0,0,0,0]:
        obj_visual_shape = bullet_client.createVisualShape(
            shapeType=bullet_client.GEOM_MESH,
            fileName=obj_file,
            meshScale=scale_factor,
            rgbaColor=rgba_color
        )
    else:
        obj_visual_shape = bullet_client.createVisualShape(
            shapeType=bullet_client.GEOM_MESH,
            fileName=obj_file,
            meshScale=scale_factor,
            rgbaColor=rgba_color
        )
    obj_orientation_quat = bullet_client.getQuaternionFromEuler(
        [math.radians(random_val_continuous(angle)) for angle in orientation]
    )
    obj_id = bullet_client.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=obj_collision_shape,
        baseVisualShapeIndex=obj_visual_shape,
        flags=obj_multi_body_flags,
        basePosition=position,
        baseOrientation=obj_orientation_quat
    )
    assert obj_id > -1, 'Multi body of obj could not be created'
    return obj_id


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


def calc_peg_slot_ref_pos(peg_bottom=0.07, slot_top=0.09, calll=''):
    peg_pos, peg_orn = self.bullet_client.getBasePositionAndOrientation(self.peg)
    peg_bottom_point = placement_utils.calc_rigid_body_coors(self.bullet_client, self.peg, peg_bottom)
    slot_pos, slot_orn = self.bullet_client.getBasePositionAndOrientation(self.slot)
    # slot_top_point = placement_utils.calc_rigid_body_coors(self.bullet_client, self.slot, slot_top)

    peg_pitch, peg_roll, peg_yaw = self.bullet_client.getEulerFromQuaternion(peg_orn)  # pitch --> pi, roll --> 0
    slot_pitch, slot_roll, slot_yaw = self.bullet_client.getEulerFromQuaternion(slot_orn)
    # print('peg pitch / slot pitch {} / {}, peg roll / slot roll {} / {}'.format(peg_pitch-math.pi, slot_pitch, peg_roll, slot_roll))
    cylinder_slot_in_base = [0.017, -0.017, 0]
    slot_top_point = np.array(slot_pos) + placement_utils.rotate_vector(cylinder_slot_in_base, slot_orn) + np.array(
        [0, 0, slot_top])
    d_q = (math.tanh(peg_pitch - math.pi) ** 2 + math.tanh(peg_roll) ** 2)  # (math.cos(peg_yaw - slot_yaw) + 1)/2
    px, py, pz = peg_bottom_point
    sx, sy, sz = slot_top_point
    d_xy = math.sqrt((sx - px) ** 2 + (sy - py) ** 2)
    d_z = pz - sz
    # print('call {}, pz {}, sz {}, dz {}'.format(calll, pz, sz, d_z))
    d_xyz = math.sqrt(d_xy ** 2 + d_z ** 2)
    # self.bullet_client.addUserDebugLine(slot_top_point, slot_top_point + np.array([0,0,0.01]), lineColorRGB=[1, 0, 0], lineWidth=3.0)
    # print('dxy {:+.3f}, dxyz {:+.3f}, dz {:+.3f}, dq {:+.3f}'.format(d_xy, d_xyz, d_z, d_q))
    return d_xy, d_xyz, d_z, d_q