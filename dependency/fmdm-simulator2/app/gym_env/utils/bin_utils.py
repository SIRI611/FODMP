import os
import math
from gym_env.utils import object_utils


def create_bin_constraint(bullet_client, bin_id, bin_config):
    """
    Creates a constraint out of a bin to allow for shaking, moving, etc the bin.

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    bin_id : int
        Unique id of bin to create a constraint out of.
    bin_config : dict
        Dictionary corresponding to the scene > bin or scene > drop-bin section of the config file for the bin
        to create a constraint out of.
    Returns
    -------
    int
        The unique id of the constraint.
    """
    use_base, no_body = -1, -1
    use_origin, upright_orientation_quat = [0, 0, 0], [0, 0, 0, 1]
    bin_orientation_quat = bullet_client.getQuaternionFromEuler(
        [math.radians(angle) for angle in bin_config['orientation']]
    )
    constraint_id = bullet_client.createConstraint(
        parentBodyUniqueId=bin_id,
        parentLinkIndex=use_base,
        childBodyUniqueId=no_body,
        childLinkIndex=use_base,
        jointType=bullet_client.JOINT_FIXED,
        jointAxis=use_origin,
        parentFramePosition=use_origin,
        childFramePosition=bin_config['position'],
        parentFrameOrientation=upright_orientation_quat,
        childFrameOrientation=bin_orientation_quat
    )
    assert constraint_id > -1, 'Constraint could not be created out of bin.'
    return constraint_id


def load_bin(bullet_client, bin_config, ml_mode, rgba_color=None):
    """
    Loads a bin according to the bin configuration passed to this function. Configurable aspects of the bin are the
    wavefront OBJ file to load, the bin's position, the bin's orientation, and the factor to scale the bin's size by.
    The color of the bin can also be specified with the rgba_color parameter.

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    bin_config : dict
        Dictionary corresponding to the scene > bin or scene > drop-bin section of the config file for the bin to load.
    ml_mode : string
        Either 'training' or 'testing'. Determines the directory a random obj file is selected from if scene > bin >
        load-random-obj-file is False (if ml_mode is 'training', obj file will be randomly selected from assets >
        training_objs and if ml_mode is 'testing', obj file will be randomly selector from assets > testing_objs). 
    rgba_color : list
        The color of the bin in RGBA format. Is dark grey by default. List of length 4. 
    Returns
    -------
    tuple
        A tuple consisting of an integer and either a list of integers (if the bin has an objects subsection in the
        config file) or a None value (if the bin does not have an objects subsection). The integer corresponds to the
        unique object id of the bin and the list of integers correspond to the unique object ids of each object spawned
        into the simulation. The list has a length equal to a random integer between the min and max number of objects
        specified in the scene > bin > objects section of the bin_config (the config file for this bin).
    """
    if rgba_color is None:
        rgba_color = [0.3, 0.3, 0.3, 1]
    fixed_object_mass = 0
    object_ids = []
    bin_file = os.environ['path-to-bins'] + bin_config['bin-file']
    bin_flags = bullet_client.GEOM_FORCE_CONCAVE_TRIMESH
    multi_body_flags = bullet_client.URDF_ENABLE_SLEEPING
    bin_orientation_quat = bullet_client.getQuaternionFromEuler(
        [math.radians(angle) for angle in bin_config['orientation']]
    )
    collision_shape = bullet_client.createCollisionShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=bin_file,
        meshScale=bin_config['scale-factor'],
        flags=bin_flags
    )
    visual_shape = bullet_client.createVisualShape(
        shapeType=bullet_client.GEOM_MESH,
        fileName=bin_file,
        meshScale=bin_config['scale-factor'],
        rgbaColor=rgba_color
    )
    bin_id = bullet_client.createMultiBody(
        baseMass=fixed_object_mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=bin_config['position'],
        baseOrientation=bin_orientation_quat,
        flags=multi_body_flags
    )
    assert bin_id > -1, 'Multi body for bin could not be created.'
    if 'objects' in bin_config:
        object_ids = object_utils.spawn_objects(bullet_client, bin_config, ml_mode)
    return bin_id, object_ids
