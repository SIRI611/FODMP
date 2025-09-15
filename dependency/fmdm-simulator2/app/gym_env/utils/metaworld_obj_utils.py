import os
import math
import numpy as np
import random
from gym_env.utils.helpers import random_val_continuous, random_val_discreete
import transforms3d.quaternions as quat_transform
from gym_env.utils.scene_utils import new_scene_obj_list, meta_world_scene_object_list


BASE=[1,-1.,0.07 ]
BASE_VEL=[ -2,2,0 ]
def continous_action_to_discrete_index(continuous_val):
    # action_mappings = [-0.75, -0.25, 0.25, 0.75]
    #                    0       1    2     3
    # [-1,-0.5)  0
    # [-0.5,0)   1
    # [0,0.5)    2
    # [0.5,1]    3
    if continuous_val<-0.5:
        return 0
    elif continuous_val<0:
        return 1
    elif continuous_val<0.5:
        return 2
    else:
        return 3  # current version does not have hand over skill. this is only used for two robots.

def generate_random_obj_pos(base_position, num_pos=2, xy_bounds=[0.5,0.5]):
    x, y, z = base_position
    dx = [-0.5, -0.7, -0.6, -0.6]  #np.random.uniform(-xy_bounds[0], xy_bounds[1], num_pos)
    dy = [0.3, -0.2, 0.3, -0.2]# np.random.uniform(-xy_bounds[0], xy_bounds[1], num_pos)
    pos_array = []
    pos_array_rand = []
    for i in range(num_pos):
        pos_array_rand.append([x+random.uniform(-0.5, 0.5), y+random.uniform(-0.5, 0.5), z])
        pos_array.append([x, y, z])
    return np.array(pos_array_rand), np.array(pos_array)

def generate_random_obj_vel(base_velocity, num_vel=1, xy_bounds=[0.5,0.5]):
    x, y, z = base_velocity
    dx = [-0.5, -0.7, -0.6, -0.6]  #np.random.uniform(-xy_bounds[0], xy_bounds[1], num_pos)
    dy = [0.3, -0.2, 0.3, -0.2]# np.random.uniform(-xy_bounds[0], xy_bounds[1], num_pos)
    vel_array = []
    vel_array_rand = []
    for i in range(num_vel):
        vel_array_rand.append([x+random.uniform(-0.5, 0.5), y+random.uniform(-0.3, 0.3), z])
        vel_array.append([x, y, z])
    return np.array(vel_array_rand), np.array(vel_array)

def reset_target_disruptive_objects(bullet_client, target_obj_infos, disrpt_obj_infos):
    random_obj_pos, original_pos = generate_random_obj_pos(BASE)
    random_obj_vel, original_vel = generate_random_obj_vel(BASE_VEL)
    pos_indx = [i for i in range(2)]
    random.shuffle(pos_indx)
    i_count = 0
    for obj in target_obj_infos:
        reset_object(bullet_client, obj['obj_id'], original_pos[pos_indx[i_count]], random_obj_vel, obj['orientation'])
        # # print(random_obj_vel)
        # bullet_client.resetBaseVelocity(
        #     obj['obj_id'],
        #     linearVelocity=random_obj_vel[0],
        #     angularVelocity=[0.0, 0.0, 0.0]
        # )
        # i_count +=1
    # for disrpt_obj in disrpt_obj_infos:
    #     reset_object(bullet_client, disrpt_obj['obj_id'], original_pos[pos_indx[i_count]], random_obj_vel, disrpt_obj['orientation'])
    #     i_count += 1

def reset_object(bullet_client, object_id, original_pos, random_vel, orn=None):
    if orn == None:
        _, obj_orientation_quat = bullet_client.getBasePositionAndOrientation(object_id)
    else:
        obj_orientation_quat = bullet_client.getQuaternionFromEuler([math.radians(i) for i in orn])
    pos = [original_pos[0]+random.uniform(-0.5, 0.5), original_pos[1]+random.uniform(-0.05, 0.05), original_pos[2]]
    linearVelocity=random_vel[0]
    # print(linearVelocity)
    bullet_client.resetBasePositionAndOrientation(object_id, pos, obj_orientation_quat)
    bullet_client.resetBaseVelocity(
            object_id,
            linearVelocity=linearVelocity,
            angularVelocity=[0.0, 0.0, 0.0]
        )
def remove_objects(bullet_client, object_list):
    for obj in object_list:
        print(bullet_client.removeBody(obj['obj_id']))

def load_target_and_disruptive_objects(bullet_client, target_object_configs, disruptive_object_configs, target_color=[1,1,1,1], disruptive_color=[0.43921569, 0.38823529, 0.25098039, 1]):
    target_obj_infos = []
    disruptive_obj_infos = []
    # generate 4 random pos
    random_obj_pos, original_pos = generate_random_obj_pos(BASE)
    random_obj_vel, original_obj_vel = generate_random_obj_vel(BASE_VEL)
    pos_indx = [i for i in range(2)]
    random.shuffle(pos_indx)
    # num_target_objs=random.randint(1,4)
    # num_disruptive_objects = 4-num_target_objs
    num_target_objs = target_object_configs['num-of-objects'][0]
    num_disruptive_objects = disruptive_object_configs['num-of-objects'][0]
    # print('num target obj {}, num disruptive obj {}'.format(num_target_objs, num_disruptive_objects))
    target_geometry = target_object_configs['geometry'][0]
    # print(target_object_configs)
    # print(disruptive_object_configs)
    for i in range(num_target_objs):
        target_obj_info = load_one_object(bullet_client, target_geometry, random_obj_pos[pos_indx[i]],
                                          original_pos[pos_indx[i]], target_object_configs['orientation'][0],
                                          target_object_configs['dimensions'][0],
                                          target_object_configs['scale-factor'][0],
                                          target_object_configs['collision-scale'][0],
                                          target_object_configs['object-mass'][0],
                                          color=target_color)
        # set velocity -> set random velocity -> wrap up this function
        ball_id = target_obj_info['obj_id']
        bullet_client.resetBaseVelocity(
            ball_id,
            linearVelocity=random_obj_vel[0],   # m/s in +x (change as you like)
            angularVelocity=[0.0, 0.0, 0.0]
        )
        target_obj_infos.append(target_obj_info)
    # disrpt_obj_geometry = disruptive_object_configs['geometry'][0]
    # for i in range(num_disruptive_objects):
    #     disruptive_obj_info = load_one_object(bullet_client, disrpt_obj_geometry,
    #                                           random_obj_pos[pos_indx[num_target_objs + i]],
    #                                           original_pos[pos_indx[num_target_objs + i]],
    #                                           disruptive_object_configs['orientation'][0],
    #                                           disruptive_object_configs['dimensions'][0],
    #                                           disruptive_object_configs['scale-factor'][0],
    #                                           disruptive_object_configs['collision-scale'][0],
    #                                           disruptive_object_configs['object-mass'][0], color=disruptive_color)
    #     disruptive_obj_infos.append(disruptive_obj_info)
    # print(target_obj_infos)
    return target_obj_infos, disruptive_obj_infos


def load_one_object(bullet_client, geometry, random_obj_pos, pos_original, obj_orientation, dimension, obj_scale, collision_scale, obj_mass, color=None):
    obj_info = {}
    obj_file = os.environ['path-to-assets'] + meta_world_scene_object_list[geometry]

    obj_id = load_single_object(bullet_client, obj_file, obj_scale,
                                obj_mass, random_obj_pos,
                                obj_orientation)
    obj_info['positions'] = pos_original
    obj_info['orientation'] = obj_orientation
    obj_info['obj_id'] = obj_id
    obj_info['dimension'] = dimension
    obj_info['geomemtry'] = geometry
    obj_info['status'] = 0
    return obj_info

def load_drawer(bullet_client, container_configs, container_idx=0):
    fixed_object_mass = 0
    slot_rgba_color = [1, 1, 1, 0]
    # geometry = random_val_discreete(container_configs['geometry'])
    geometry = container_configs['geometry'][container_idx]  # test trash Can
    scale_factor = container_configs['scale-factor'][container_idx]
    container_flags = bullet_client.GEOM_FORCE_CONCAVE_TRIMESH
    container_multi_body_flags = bullet_client.URDF_ENABLE_SLEEPING
    container_position =container_configs['position'][container_idx]
    container_orientation_euler = container_configs['orientation'][container_idx]
    container_orientation_quat = bullet_client.getQuaternionFromEuler([math.radians(i) for i in container_orientation_euler])
    container_id = bullet_client.loadURDF(
        os.path.join(os.environ['path-to-assets'], meta_world_scene_object_list[geometry]),
        basePosition=container_position,
        baseOrientation=container_orientation_quat,
        globalScaling=scale_factor[0]
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
    # print(obj_file)
    obj_multi_body_flags = bullet_client.URDF_ENABLE_SLEEPING
    obj_orientation_quat = bullet_client.getQuaternionFromEuler(
        [math.radians(random_val_continuous(angle)) for angle in orientation]
    )
    obj_id = bullet_client.loadURDF(
        os.path.join(os.environ['path-to-assets'], obj_file),
        basePosition=position,
        baseOrientation=obj_orientation_quat,
        globalScaling=scale_factor[0]
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

def AB_line_dist(A, B, dist, fromA_B=0):
    ax, ay, _ = A
    bx, by, _ = B
    if (ax - bx) < 0.001:
        slope = math.pi/2
    else:
        slope = math.atan((float(by-ay))/(bx-ax))
    if fromA_B == 0:
        dist_x = ax + dist * math.cos(slope)
        dist_y = ay + dist * math.sin(slope)
    else:
        dist_x = bx + dist * math.cos(slope)
        dist_y = by + dist * math.sin(slope)
    return dist_x, dist_y

