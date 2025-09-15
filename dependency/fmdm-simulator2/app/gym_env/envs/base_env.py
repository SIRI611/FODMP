import yaml
import gym
import pkgutil
import pybullet
from gym.utils import seeding
from pybullet_utils import bullet_client as bc
from gym_env.components.camera import Camera
from gym_env.utils import scene_utils


class BaseGymEnv(gym.Env):
    """
    Base robotic manipulation custom gym environment.
    A subclass must implement the following abstract functions according to the desired functionality:
        check_termination(..)
        get_reward(..)
        get_state(..)
        set_space_attributes(..)
        step(..)
    A subclass should also implement a reset(..) method to reset any class specific attributes that exist in the
    simulation. 
    The following attributes must also be set in a subclass (the idea is for them to be set in the abstract function
    set_space_attributes(..)):
        observation_space
        action_space
        reward_range
    """
    metadata = {'render.modes': ['human', 'rgb_array', 'rgb_and_depth_arrays']}

    def __init__(self, config_file):
        """
        Initializes a base gym environment (though the gym interface is not fully implemented in this class - see
        subclases) by parsing config file, setting necessary parameters, and initializing a Bullet client.
        Note: nothing is loaded into the Bullet client until reset is called.

        Parameters
        ----------
        config_file : str
            Path to yaml file consisting of simulation configuration.
        """
        self.np_random, self.cameras = None, None
        self.action_space, self.observation_space, self.reward_range = None, None, None
        self.physics_client_id = -1
        self.config_file = self.configure(config_file=config_file)
        self.sim_mode = self.config_file['simulation']['sim-mode']
        self.ml_mode = self.config_file['simulation']['ml-mode']
        self.render_mode = self.config_file['simulation']['render-mode']
        self.bullet_client, self.plugin = self.initialize_client()

    def close(self):
        """
        Performs any necessary cleanup after the simulation has completed.
        """
        if self.sim_mode == 'gpu-headless':
            self.bullet_client.unloadPlugin(self.plugin)
        self.physics_client_id = -1
        self.bullet_client.disconnect()

    def configure(self, config_file):
        """
        Opens and reads the config file and returns the data in dictionary form.

        Parameters
        ----------
        config_file : str
            Path to yaml file consisting of simulation configuration.
        Returns
        -------
        dict
            The config file data as a dictionary.
        """
        simulator_config = {}
        with open(config_file) as config:
            try:
                simulator_config = yaml.safe_load(config)
            except Exception as err:
                print('Error Configuration File:{}'.format(err))
                raise err
        return simulator_config

    def initialize_client(self):
        """
        Gets a Bullet client instance via GUI or headless connection and stores it as an instance attribute.

        Returns
        -------
        tuple
            The first element of the tuple is an instance of pybullet_utils.bullet_client.BulletClient, which is a
            Pybullet client. The second element of the tuple is either the unique id of the EGL plugin if
            sim_mode='gpu-headless', otherwise it is None.
        """
        bullet_client, plugin_id = None, None
        allowed_sim_modes = ['gpu-gui', 'gpu-headless', 'cpu-headless'] 
        assert self.sim_mode in allowed_sim_modes, 'Please specify a valid simulation mode in the config file, ' \
                                                   'one of {}'.format(allowed_sim_modes)
        if self.sim_mode == 'gpu-gui':
            bullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        elif self.sim_mode == 'gpu-headless':
            bullet_client, plugin_id = self.load_egl_plugin()
        elif self.sim_mode == 'cpu-headless':
            bullet_client = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.physics_client_id = bullet_client._client
        return bullet_client, plugin_id

    def load_egl_plugin(self):
        """
        Loads the EGL plugin which allows the simulator to use the OpenGL renderer without an X11 context.

        Returns
        -------
        tuple
            The first element of the tuple is an instance of pybullet_utils.bullet_client.BulletClient, which is a
            Pybullet client. The second element of the tuple is the unique id of the EGL plugin.
        """
        try:
            egl = pkgutil.get_loader('eglRenderer')
            bullet_client = bc.BulletClient(connection_mode=pybullet.DIRECT)
            plugin_id = bullet_client.loadPlugin(egl.get_filename(), '_eglRendererPlugin')
            if plugin_id < 0:
                raise Exception('The plugin was not loaded properly.')
            else:
                return bullet_client, plugin_id
        except Exception as ex1:
            self.close()
            print(type(ex1))
            print(ex1.args[0])

    def render(self, mode=None):
        """
        Renders the current state of the environment.

        Parameters
        ----------
        mode : str
            The desired render mode (i.e. human or rgb).
        Returns
        -------
        any
            Depends on the value of the mode parameter. If mode='rgb_and_depth_arrays' returns a tuple of numpy arrays
            where the first item is the rgb render data (shape is height_resolution x width_resolution x 3) and the
            second item is the depth render data (shape is height_resolution x width_resolution). If mode='rgb_array' 
            returns a numpy array of the rgb render data (shape is height_resolution x width_resolution x 3). If
            mode='human' returns None.
        """
        if mode is None:
            mode = self.render_mode
        return tuple( camera.render(self.sim_mode, mode) for camera in self.cameras )

    def init_scene(self):
        self.bullet_client.resetSimulation()
        scene_utils.set_physics_engine_parameters(self.bullet_client, self.config_file['simulation'])
        scene_utils.set_view_camera(self.bullet_client, self.config_file['simulation']['default_view_cam'])
        if (self.sim_mode == 'gpu-gui') or (self.sim_mode == 'gpu-headless'):
            # scene_utils.disable_rendering_in_gui(self.bullet_client)
            scene_utils.default_rendering(self.bullet_client)
        self.reset_camera()

    def reset(self):
        """
        Removes all objects from the simulation and resets class-relevant attributes. Then respawns the scene. Also
        resets the camera view and projection matrices according to the values specified in the config file.
        Note: rendering is disabled during the loading process if gui mode is enabled (see BaseScene.reset()) in order
        to speed up the loading process.
        """

        raise NotImplementedError

    def reset_camera(self):
        """
        Resets the camera view and projection matrices according to the values specified in the config file.
        """
        self.cameras = tuple( base_env.Camera(camera_config, self.bullet_client) for camera_config in self.config_file['cameras'] )

    def seed(self, seed=None):
        """
        Sets the seed for the environments rng.

        Parameters
        ----------
        seed : int
            Desired seed for the environments rng, must be positive.
        Returns
        -------
        list
            The seed of the environments rng as the sole element of a list (length 1).
        """
        if seed is not None:
            seed = abs(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def check_termination(self):
        """
        Checks whether or not the simulation has reached a terminating state (method by which this is done depends on
        the implementation in a given environment and the config file).

        Returns
        -------
        tuple
            A boolean value and a dictionary. The boolean value represents whether or not the simulation is in a
            terminating state and the dictionary contains auxillary diagnostic info (useful for debugging and sometimes
            training).
        """
        raise NotImplementedError

    def get_reward(self, info):
        """
        Calculates the reward based on the current state of the agent and the environment.

        Parameters
        ----------
        info : dict

        Returns
        -------
        float
            Value of the reward.
        """
        raise NotImplementedError

    def get_state(self):
        """
        Gets the current state of the robots(s) in the environment.

        Returns
        -------
        object
            Observation of the current state of the robot(s) in the environment
        """
        raise NotImplementedError

    def set_space_attributes(self):
        """
        Defines the action space, observation space, and reward range in terms of the types provided by python package
        gym.spaces (i.e. spaces.Dict(), spaces.Box()) and sets them as instance attributes.
        """
        raise NotImplementedError

    def step(self, action):
        """
        Must be implemented in subclass. Runs one step of the simulation.

        Parameters
        ----------
        action : object
            The action to apply to the model

        Returns
        -------
        tuple
            Observation (object), reward (float), done (bool), info (dict). Where observation represents
            the current state of the environment, reward is the amount of reward for the previous action, done is
            whether or not to terminate the simulation (based on termination config), and info contains auxillary
            diagnostic info (useful for debugging and sometimes training).
        """
        raise NotImplementedError
