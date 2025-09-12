import os
import random


def get_obj_file_to_load(object_config, train):
    """
    Gets the name of the wavefront OBJ file that is to be loaded into the simulator as an object for picking
    according to the object config for the corresponding bin.

    Parameters
    ----------
    object_config : dict
        A dictionary corresponding to the scene > bin > objects section of the config file.
    train : bool
        Whether an agent is currently being trained (True) or tested (False). Determines the directory a random obj
        file is selected from if scene > bin > load-random-obj-file is False (if train, obj file will be randomly
        selected from assets > training_objs and if not train, obj file will be randomly selected from assets >
        testing_objs).
    Returns
    -------
    string
        Absolute path to the object to load.
    """
    simple_objects = ['cylinder', 'sphere']
    assert type(train) is bool, 'simulation > train must be given a bool value in the config file based on whether ' \
                                'an agent is currently being trained or tested.'
    if object_config['load-random-obj-file']:
        path = ''
        file_type = '.obj'
        if train:
            path = os.environ['path-to-training-objs']
        else:
            path = os.environ['path-to-testing-objs']
        obj_files_to_pick_from = [obj for obj in os.listdir(path) if obj[-len(file_type):] == file_type]
        return path + random.choice(obj_files_to_pick_from)
    else:
        if object_config['obj-file-to-load'] in simple_objects:
            return object_config['obj-file-to-load']
        elif train:
            return os.environ['path-to-training-objs'] + object_config['obj-file-to-load']
        else:
            return os.environ['path-to-testing-objs'] + object_config['obj-file-to-load']


def get_obj_spawn_positions(spawn_origin, number_of_objs):
    """
    Generates a list of length number_of_objs which consists of the coordinates to load each object. The objects are
    loaded with a random x coordinate (within a 3.5 cm radius of +-spawn_origin[x]) and a y coordinate that changes by
    slightly more than max diameter of an object (3.5 cm) each iteration. When the y coordinate exits the scope of the
    3.5 cm radius of +-spawn_origin[y], it is set back to its starting position (spawn_origin[y] - 3.5 cm) and the z
    coordinate is incremented by slightly more than the max diameter of of an object (3.5 cm). Note: the max dimension
    of all objects in the training_objs and testing_objs folder is guarenteed to be between 1 and 3 cm.

    Parameters
    ----------
    spawn_origin : list
        Coordinate to spawn objects around in the form [x, y, z]. Objects will be spawned within a 3.5 cm radius of
        this coordinate. List of length 3.
    number_of_objs : int
        Number of objects that are to be loaded.
    Returns
    -------
    list
        List consisting of sublists in the form of [x, y, z] which correspond to coordinates of each object that is to
        be loaded. List of length number_of_objs.
    """
    drop_radius, max_obj_radius, z_offset, z_step = 0.035, 0.035, 0.02, 0.03
    x, y, z = 0, 1, 2
    min_drop_x, max_drop_x = spawn_origin[x] - drop_radius, spawn_origin[x] + drop_radius
    drop_location_y = spawn_origin[y] - drop_radius
    drop_location_z = spawn_origin[z] + z_offset
    coords_of_all_objects = []
    for _ in range(number_of_objs):
        coords_of_all_objects.append([random.uniform(min_drop_x, max_drop_x), drop_location_y, drop_location_z])
        if (drop_location_y + max_obj_radius) >= spawn_origin[y] + drop_radius:
            drop_location_y = spawn_origin[y] - drop_radius
            drop_location_z = drop_location_z + z_step
        else:
            drop_location_y += max_obj_radius
    return coords_of_all_objects


def random_val_continuous(arg):
    """
    Will check if a value should be random, and randoimize it if so.

    Parameters
    ----------
    arg : list or float
        The value to possibly be randomized
    Returns
    -------
    float
        If arg is a list it is a random number between arg[0] and arg[1],
        otherwise it is arg.
    """
    if isinstance(arg,list):
        return random.uniform(*arg)
    else:
        return arg

def random_val_discreete(arg):
    """
    Will check if a value should be random, and randoimize it if so.

    Parameters
    ----------
    arg : list or any
        The value to possibly be randomized
    Returns
    -------
    any
        If arg is a list it is a random selection from arg,
        otherwise it is arg.
    """
    if isinstance(arg,list):
        return random.choice(arg)
    else:
        return arg

def calc_up_vector(quaternion_vec):
    from pyquaternion import Quaternion
    my_q = Quaternion(quaternion_vec)
    return my_q.rotate([0,0,1])
