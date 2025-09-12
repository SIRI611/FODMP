import os

def default_rendering(bullet_client):
    bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_PLANAR_REFLECTION, 1)
    # GUI, shadows, and rendering are enabled by default - uncomment the respective line of code below to disable
    # GUI (x, y, z axis and the camera images/view), shadows, or rendering
    bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_GUI, 0)
    bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    # disable tinyrenderer, software (CPU) renderer, we don't use it here
    bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_SHADOWS, 0)
    # self.bullet_client.configureDebugVisualizer(self.bullet_client.COV_ENABLE_RENDERING, 1)
    # self.bullet_client.configureDebugVisualizer(self.bullet_client.COV_ENABLE_GUI, 1)


def disable_rendering_in_gui(bullet_client):
    """
    Configures the OpenGL renderer when in GUI mode by disabling rendering and all rendering features/preivews. It is
    recommended to call this function prior to loading any robots or objects in order to drastically speed up the
    loading process. Other configurations can also be applied - see the pybullet quick start guide for more options.

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    """
    print('disable rendering in GUI')
    disable = 0
    bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_RENDERING, disable)
    bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_GUI, disable)
    bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_RGB_BUFFER_PREVIEW, disable)
    bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW, disable)
    bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, disable)
    bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_SHADOWS, disable)
    bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_PLANAR_REFLECTION, disable)


def enable_rendering_in_gui(bullet_client, enable_windows):
    """
    Configures the OpenGL renderer when in GUI mode by enabling rendering and previews of the rgb and depth buffer
    depending on the value of the parameter enable_windows. Other configurations can also be applied - see the pybullet
    quick start guide for more options.

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    enable_windows : bool
        Whether to make additional features visible in the GUI, including lines corresponding to the x, y, and z axis
        and previews of the rgb buffer and depth buffer. If a preview of the segmentation mark is also desired then
        it must also be enabled below.
    """
    enable = 1
    bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_RENDERING, enable)
    bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_SHADOWS, 0)
    if enable_windows:
        bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_GUI, enable)
        bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_RGB_BUFFER_PREVIEW, enable)
        bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW, enable)


def load_plane(bullet_client, path_to_plane_urdf, rgba_color=None):
    """
    Loads a urdf representing a plane (or grid) into the given pybullet client.

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    path_to_plane_urdf : string
        Absolute path to the urdf file representing the plane (or grid) to load into the simulation.
    rgba_color : list
        The color of the plane (or grid) in RGBA format, list of length 4.
    Returns
    -------
    int
        Unique id of the plane (or grid). 
    """
    if rgba_color is None:
        rgba_color = [1, 1, 1, 0.8]
    apply_to_base = -1
    flags = bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | bullet_client.URDF_ENABLE_SLEEPING
    ground_id = bullet_client.loadURDF(
        fileName=path_to_plane_urdf,
        basePosition=[0,0,-0.625],
        flags=flags
    )
    assert ground_id > -1, 'The plane urdf could not be loaded into the simulation.'
    bullet_client.changeVisualShape(ground_id, apply_to_base, rgbaColor=rgba_color)
    return ground_id

def load_table(bullet_client, rgba_color=None):
    """
    Loads a table into the given pybullet client.

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    rgba_color : list
        The color of the table in RGBA format, list of length 4.
        If unset the color of the table will be unchanged.
    Returns
    -------
    tuple
        A tuple consisting of an int and a float. The int is the unique id of the table and
        the float is the vertiacal position of the top surface of the table.
    """
    import pybullet_data
    flags = bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | bullet_client.URDF_ENABLE_SLEEPING
    table_id = bullet_client.loadURDF(
        fileName=os.environ['path-to-table'] + "table.urdf",
        basePosition=[1,0,-0.625],
        flags=flags
    )
    assert table_id > -1, 'The plane urdf could not be loaded into the simulation.'
    if rgba_color != None:
        apply_to_base = -1
        bullet_client.changeVisualShape(table_id, apply_to_base, rgbaColor=rgba_color)
    _,_,_,table_dim,_,table_pos,_ = bullet_client.getCollisionShapeData(table_id,-1)[0]
    table_height = table_pos[2] + table_dim[2]/2
            # set friction for table
    bullet_client.changeDynamics(table_id, -1,
                                  lateralFriction=0.4,# default is 0.5 ~ 1.0
                                  spinningFriction=0.0,
                                  rollingFriction=0.0)
    return table_id, table_height

def set_physics_engine_parameters(bullet_client, simulation_config):
    """
    Sets the physics engine parameters of a given pybullet client as specified in a given config file (in the form of
    a dict). This function should only be called when the simulator is initially being created or after the simulator
    has been reset, as setting the physics engine parameters too often can break the physics engine.

    Parameters
    ----------
    bullet_client : pybullet_utils.bullet_client.BulletClient
        Pybullet client.
    simulation_config : dict
        Dictionary containing the simulation section of the configuration file for the simulator.
    """
    bullet_client.setRealTimeSimulation(0)
    bullet_client.setGravity(0, 0, -simulation_config['gravity'])
    bullet_client.setPhysicsEngineParameter(
        fixedTimeStep=simulation_config['timestep'],
        numSolverIterations=simulation_config['number-of-solver-iterations'],
        numSubSteps=simulation_config['number-of-sub-steps'],
        contactERP=simulation_config['contact-erp']
    )


def set_view_camera(bullet_client, view_camera_config):
    bullet_client.resetDebugVisualizerCamera(view_camera_config['distance'],
                                             view_camera_config['yaw'],
                                             view_camera_config['pitch'],
                                             view_camera_config['target'])



new_scene_obj_list={
"bolt":	"bolt.obj",
"CoffeeCup": "coffee_cup_obj.obj",
"CookTop": "11632_Cooktop_v2_L3.obj",
"CookTop2": "11633_Cooktop_v1_L3.obj",
"donut": "donut.obj",
"Frying_Pan": "10262_12_Frying_Pan_max2011_v2_it2.obj",
"hammer":"10293_Hammer_v1_iterations-2.obj",
"kettle":"20901_Whistling_Tea_Kettle_v1.obj",
"plate":"plate.obj",
"sander":"17732_Random_Orbital_Sander_v1-1_NEW.obj",
"SodaCan":"14025_Soda_Can_v3_l3.obj",
"Table":"table.obj",
"TrashCan":"trashcan.obj",
"WoodenBox":"wooden crate.obj",
"mug":"mug.obj",
"pallet":"pallet.obj",
"spoon":"spoon.obj",
"wrench":"Wrench.obj",
"bins":"bin0.obj",
"capacitor":"capacitor_12x12x27mm.obj",
"pegs": ["cylinder.obj","cube_leg.obj","triangle.obj"],
"slots": ["cylinder.obj", "cube_leg.obj", "triangle.obj"],
"fork": "10290_Fork_v2_iterations-2.obj",
"connectors": ["D-SUB.obj", "Ethernet Coupler.obj"],
"usb": "usb.obj"
}

meta_world_scene_object_list={
    'cup': 'assets_v2/objects/mug.urdf',
    'goal': 'assets_v2/objects/puck_goal.urdf',
    'soccerball': "soccerball/soccerball.urdf",
    'drawer': "assets_v2/objects/drawer.urdf",
    'hammer': "assets_v2/objects/hammer.urdf",
    'drawer_closed': "assets_v2/objects/drawer_closed.urdf"
}
