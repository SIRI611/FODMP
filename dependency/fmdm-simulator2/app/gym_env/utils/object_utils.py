import random
import numpy as np
from gym_env.utils import helpers


def allow_objects_to_fall(bullet_client, steps_for_objects_to_fall=250):
    """
    After all objects have been loaded, the physics engine is stepped steps_for_objects_to_fall (as specified in the
    config file) number of times in order to allow the objects to fall within the bin.

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    steps_for_objects_to_fall : int
        Number of times the physics engine should be stepped to allow for objects to fall into place.
    """
    for _ in range(steps_for_objects_to_fall):
        bullet_client.stepSimulation()


def remove_object(bullet_client, object_id):
    """
    Removes an object from the simulation.

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    object_id : int
        Unique id of object to remove.
    """
    bullet_client.removeBody(object_id)


def remove_objects_not_in_wksp(bullet_client, objects, lower_wksp_bound, upper_wksp_bound):
    """
    Removes all objects that are not within the specified workspace from the simulation and removes their unique ids
    from the objects list passed to this function as a parameter. Returns the updated list of object ids that
    correspond to the objects that are within the specified workspace and thus still in the simulation.

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    objects : list
        A list of integers corresponding to the unique ids of objects to check if in workspace.
    lower_wksp_bound : numpy.ndarray
        Array with shape (3,) consisting of the lower bounds of the workspace in the x, y, and z dimension.
    upper_wksp_bound : numpy.ndarray
        Array with shape (3,) consisting of the upper bounds of the workspace in the x, y, and z dimension.
    Returns
    -------
    list
        A list consisting of the unique ids of the objects that are within the workspace and thus still in the
        simulation.
    """
    position_index = 0
    objects_to_remove = []
    for obj_id in objects:
        original_obj_position = np.array(bullet_client.getBasePositionAndOrientation(obj_id)[position_index])
        clipped_obj_position = np.clip(original_obj_position, a_min=lower_wksp_bound, a_max=upper_wksp_bound)
        if (original_obj_position != clipped_obj_position).any():
            bullet_client.removeBody(obj_id)
            objects_to_remove.append(obj_id)
    for obj_id in objects_to_remove:
        objects.remove(obj_id)
    return objects


def spawn_objects(bullet_client, bin_config, train):
    """
    Spawns objects according to the configuration specified by the parameter bin_config which is the scene > bin section
    section of the config file.

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    bin_config : dict
        Dictionary corresponding to the scene > bin or scene > drop-bin section of the config file for the bin
        to create a constraint out of.
    train : bool
        Whether an agent is currently being trained (True) or tested (False). Determines the directory a random obj
        file is selected from if scene > bin > load-random-obj-file is False (if train, obj file will be randomly
        selected from assets > training_objs and if not train, obj file will be randomly selected from assets >
        testing_objs).
    Returns
    -------
    list
        A list of integers corresponding to the unique ids of each object spawned into the simulation. Has a length
        equal to a random integer between the min and max number of objects specified in the scene > bin > objects
        section of bin_config (the config file for the corresponding bin).
    """
    default_obj_mass = 0.1
    light_gray = [0.8, 0.9, 1, 1]
    flags = bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | bullet_client.URDF_ENABLE_SLEEPING
    number_of_objs = random.randint(
        bin_config['objects']['min-number-of-objects'],
        bin_config['objects']['max-number-of-objects']
    )
    obj_spawn_positions = helpers.get_obj_spawn_positions(bin_config['position'], number_of_objs)
    obj_file_to_load = helpers.get_obj_file_to_load(bin_config['objects'], train)
    if obj_file_to_load == 'cylinder':
        collision_shape = bullet_client.createCollisionShape(
            shapeType=bullet_client.GEOM_CYLINDER,
            radius=0.0075,
            height=0.02,
            meshScale=bin_config['objects']['scale-factor'],
            collisionFramePosition=[0, 0, 0]
        )
        visual_shape = bullet_client.createVisualShape(
            shapeType=bullet_client.GEOM_CYLINDER,
            radius=0.0075,
            length=0.02,
            meshScale=bin_config['objects']['scale-factor'],
            rgbaColor=light_gray,
            visualFramePosition=[0, 0, 0]
        )
    elif obj_file_to_load == 'sphere':
        collision_shape = bullet_client.createCollisionShape(
            shapeType=bullet_client.GEOM_SPHERE,
            radius=0.0075,
            meshScale=bin_config['objects']['scale-factor'],
            collisionFramePosition=[0, 0, 0]
        )
        visual_shape = bullet_client.createVisualShape(
            shapeType=bullet_client.GEOM_SPHERE,
            radius=0.0075,
            meshScale=bin_config['objects']['scale-factor'],
            rgbaColor=light_gray,
            visualFramePosition=[0, 0, 0]
        )
    else:
        collision_shape = bullet_client.createCollisionShape(
            shapeType=bullet_client.GEOM_MESH,
            fileName=obj_file_to_load,
            meshScale=bin_config['objects']['scale-factor']
        )
        visual_shape = bullet_client.createVisualShape(
            shapeType=bullet_client.GEOM_MESH,
            fileName=obj_file_to_load,
            meshScale=bin_config['objects']['scale-factor'],
            rgbaColor=light_gray
        )
    assert collision_shape > -1, 'Collision shape of object could not be created.'
    assert visual_shape > -1, 'Visual shape of object could not be created.'
    object_ids = list(bullet_client.createMultiBody(
        baseMass=default_obj_mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        flags=flags,
        batchPositions=obj_spawn_positions
    ))
    return object_ids