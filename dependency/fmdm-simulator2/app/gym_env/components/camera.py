import numpy as np
import math

class Camera:
    """
    A class for rendering a pybullet environment. Depending on configs, different render settings will be used.
    """
    render_modes = ['human', 'rgb_array', 'rgb_and_depth_arrays']

    def __init__(self, camera_config, bullet_client, camera_name=None):
        """
        Initializes a camera object with the configuration data.

        Parameters
        ----------
        camera_config : dict
            Camera configuration data.
        bullet_client : pybullet_utils.bullet_client.BulletClient
            Pybullet client.
        """
        self.bullet_client = bullet_client
        self.scale_depth_data = camera_config['scale-depth-data']
        self.render_width = camera_config['width-resolution']
        self.render_height = camera_config['height-resolution']
        self.eye_position = camera_config['view-matrix']['eye-position']
        self.target_position = camera_config['view-matrix']['target-position']
        self.up_vector = camera_config['view-matrix']['up-vector']
        self.fov = camera_config['projection-matrix']['fov']
        self.near_plane = camera_config['projection-matrix']['near-plane']
        self.far_plane = camera_config['projection-matrix']['far-plane']
        self.fx_inv =  2 * math.tan(self.fov * np.pi / 360) / self.render_width
        self.fy_inv =  2 * math.tan(self.fov * np.pi / 360) / self.render_height
        self.cx = self.render_width / 2
        self.cy = self.render_height / 2
        aspect = self.render_width / self.render_height
        if camera_config['load_mode'] == 'up_vector':
            self.view_matrix = self.bullet_client.computeViewMatrix(
                cameraEyePosition=self.eye_position,
                cameraTargetPosition=self.target_position,
                cameraUpVector=self.up_vector
            )
        else:
            self.view_matrix = self.bullet_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.target_position,
                distance=camera_config['view-matrix']['distance'],
                yaw=self.up_vector[0],
                pitch=self.up_vector[1],
                roll=self.up_vector[2],
                upAxisIndex=2
            )  # corner
        self.projection_matrix = self.bullet_client.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=aspect,
            nearVal=self.near_plane,
            farVal=self.far_plane
        )

    def adjust_view_matrix(self, eye_position, target_position, up_vector):
        """
        Adjusts the view matrix of this camera.

        Parameters
        ----------
        eye_position : list
            The eye position of the camera in Cartesian world coordinates. List of length 3.
        target_position : list
            The focus point of the camera in Cartesian world coordinates. List of length 3.
        up_vector : list
            The up vector of the camera in Cartesian world coordinates. List of length 3
        """
        self.eye_position = eye_position
        self.target_position = target_position
        self.up_vector = up_vector
        self.view_matrix = self.bullet_client.computeViewMatrix(
            cameraEyePosition=self.eye_position,
            cameraTargetPosition=self.target_position,
            cameraUpVector=self.up_vector
        )

    def get_image(self, sim_mode, render_mode):
        """
        Gets the image via GUI render engine (OpenGL) or headless render engine (TinyRenderer). Can also return the
        camera depth data as a greyscale image if desired (set render_mode to \'rgb_and_depth_arrays\'). The true
        z-value is calculated from the depth pixels using the equation given in the documentation for
        pybullet.getCameraImage(..). The resulting array is then scaled such that it is possible for a greyscale image
        to be created from it.

        Parameters
        ----------
        sim_mode : string
            The mode of the simulation (i.e. one of 'gpu-gui', 'gpu-headless', or 'cpu-headless').
        render_mode : str
            The desired render mode (i.e. human or rgb or rgb and depth).
        Returns
        -------
        tuple
            Either a single numpy array consisting of the rgb render data (shape is height_resolution x
            width_resolution x 3) or a tuple consisting of the rgb render data and the depth render data (shape of the
            depth render data is (height_resolution x width_resolution).
        """
        allowed_sim_modes = ['gpu-gui', 'gpu-headless', 'cpu-headless'] 
        assert sim_mode in allowed_sim_modes, 'Please specify a valid simulation mode in the config file, one of ' \
                                              '{}'.format(allowed_sim_modes)
        if sim_mode == 'gpu-gui':
            (_, _, rgb_px, depth_px, seg_px) = self.bullet_client.getCameraImage(
                width=self.render_width,
                height=self.render_height,
                viewMatrix=self.view_matrix,
                projectionMatrix=self.projection_matrix,
                renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL,
                shadow = 0
            )
        elif sim_mode == 'gpu-headless':
            (_, _, rgb_px, depth_px, seg_px) = self.bullet_client.getCameraImage(
                width=self.render_width,
                height=self.render_height,
                viewMatrix=self.view_matrix,
                projectionMatrix=self.projection_matrix,
                shadow=0,
                renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL
            )
        elif sim_mode == 'cpu-headless':
            (_, _, rgb_px, depth_px, seg_px) = self.bullet_client.getCameraImage(
                width=self.render_width,
                height=self.render_height,
                viewMatrix=self.view_matrix,
                projectionMatrix=self.projection_matrix,
                shadow = 0,
                renderer=self.bullet_client.ER_TINY_RENDERER
            )
        rgb_array = np.array(rgb_px)
        rgb_array = rgb_array[:, :, :3]
        if render_mode == 'rgb_array':
            return rgb_array
        else:
            depth_px = self.far_plane * self.near_plane / (self.far_plane - ((self.far_plane - self.near_plane) * \
                                                                             np.array(depth_px)))
            if self.scale_depth_data:
                image_scale = 255
                depth_px = (depth_px * image_scale).astype(np.uint8)
            if render_mode == 'rgb_and_depth_arrays':
                return rgb_array, depth_px
            else:
                return depth_px

    def render(self, sim_mode, render_mode):
        """
        Renders the current pybullet environment. The modes 'human' and 'rgb_array' are supported and do the following:
        'human' mode returns nothing and continues to display the simulation if in GUI mode (usually for human
        consumption). 'rgb_array' returns a numpy.ndarray with shape (x, y, 3), representing RGB values for an x-by-y
        pixel image, suitable for turning into a video. Can also return the camera depth data as a greyscale image if
        desired (set render_mode to \'rgb_and_depth_arrays\').

        Parameters
        ----------
        sim_mode : string
            The mode of the simulation (i.e. one of 'gpu-gui', 'gpu-headless', or 'cpu-headless').
        render_mode : str
            The desired render mode (i.e. human or rgb or rgb and depth).
        Returns
        -------
        numpy.ndarray
            Array consisting of the rgb render data (shape is height_resolution x width_resolution x num_colors).
            Num_colors is 3 for an rgb image.
        """
        # assert render_mode in self.render_modes, 'Invalid render mode, please use one of {}.'.format(self.render_modes)
        if render_mode != 'human':
            return self.get_image(sim_mode, render_mode)
