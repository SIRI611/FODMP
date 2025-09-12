# Configuration File

The Franka Emika simulator reads its configuration from gym_env/configs/default.yaml. We use a yaml file (for more info about yaml files see [here](https://pyyaml.org/wiki/PyYAMLDocumentation)) to specify the configuration that we want the simulator to load and run. Currently the following attributes can be configured:

## simulation
*  sim-mode: string `gpu-gui` or `gpu-headless` or `cpu-headless`
> The mode that the simulation should run in. The naming convention is [cpu,gpu]-[gui,headless], with the supported options shown above. The `gpu-gui` mode will use the OpenGL renderer (runs on the gpu) with a X11 context (gui). The `gpu-headless` mode will use the OpenGL renderer (runs on the gpu) without a X11 context (headless). The `cpu-headless` mode will use the TinyRenderer renderer (runs on the cpu) in headless mode. Currently no mode that uses a cpu renderer in gui mode is supported.
*  ml-mode: string `training` or `testing`
> Whether an agent is being trained or tested (the simulation selects which folder to pick a random object from based on whether we are currently training or testing an agent)
*  render-mode: string `human` or `rgb_array` or `rgb_and_depth_arrays`
> The method by which rendering should be performed. If `human` mode is specified, calling render(..) on the environment will not return anything. If `rgb_array` mode is specified, calling render(..) will return an array consisting of the rgb camera data. If `rgb_and_depth_arrays` mode is specified, calling render(..) will return two arrays, one consisting of the rgb camera data and the other consisting of the depth camera data. For more info, see the documentation for base_env.BaseGymEnv.render(..) as well as the camera.Camera class.
*  enable-windows: bool
> Whether extra GUI features such as rgb camera and depth camera previews are enabled (if the simulation is in GUI mode)
*  gravity: float

*Note the following four physics engine parameters can cause undefined behavior if not configured properly. It is best to use the default values unless you know that the values you are specifying will result in a stable environment.*

*  timestep: float
> The amount of simulated time that is progressed each simulation step, the pybullet default is 1/240 but the **suggested value is 0.001 (approx 1/960)**
*  number-of-sub-steps: int
> The number of sub steps that each simulation step is divided into, the pybullet default is 0 but the **suggested value is 2**
*  number-of-solver-iterations: int
> Max number of constraint solver iterations, the pybullet default and **suggested value is 50**
*  contact-erp: float
> Contact error reduction parameter - pybullet recommends between 0.1 and 0.8, the default and **suggested value is 0.2**

## termination
### terminate-on-collision-with-bin
*  enable: bool
> Whether or not an arm colliding with a bin is used as a terminating condition
### terminate-after-successful-grabs
*  enable: bool
> Whether or not a specified number of successful grabs is used as a terminating condition
*  ensure-grab-is-stable: bool
> Once a normal grab has happened the program will shake the robot and then ensure that the object is still being grasped by the robot
*  grabs-before-termination: int
> If enable is True, the number of successful grabs the arm must perform before this terminating condition returns True

## camera
*Note all suggested values in this section correspond to those known to return valid depth data. In order to adjust the depth data, experiment with different projection-matrix > near-plane and projection-matrix > far-plane values.*

*  scale-depth-data: bool
> Whether to scale the camera depth data returned by Camera.render(..) by a factor of 255 (all values in the depth data array are originally between 0 and 1 and correspond to the true z-values from the depth buffer) 
*  width-resolution: int
*  height-resolution: int
### view-matrix
*  eye-position: [float, float, float]
> The eye position of the camera in Cartesian world coordinates - list is of length 3, **suggested location is 0.155m above the bin to view**
*  target-position: [float, float, float]
> The focus point of the camera in Cartesian world coordinates - list is of length 3, **suggested location is the point on the ground directly below the origin of the bin**
*  up-vector: [float, float, float]
> The up vector of the camera in Cartesian world coordinates - list is of length 3, **suggested value is [0,1,0] (returns a right-side-up image)**
### projection-matrix
*  fov: int
> Field of view, **suggested value is 76**
*  near-plane: float
> Near plane distance, **suggested value is 0.008**
*  far-plane: float
> Far plane distance, **suggested value is 0.16**

## arms
### arm-#
> This section would be repeated in configs for simulations of > 1 arm with any subsequent arms specified as arm-#
*  type: string `franka`
> Type of arm to load
*  clip-delta-actions-to-workspace: bool
> Whether to clip delta actions to the workspace (so no actions result in the end effector going through a bin or colliding with a bin for example)
*  get-end-effector-force-sensor-readings: bool
> Whether to enable force/torque sensors in each end effector and get readings of the reaction forces sensed by each end effector at each simulation step throughout the apply_action_pos or the apply_action_delta process
*  position: [float, float, float]
> Position of base of arm
*  orientation: [float, float, float]
> Orientation of base of arm (in degrees)
#### joints
*  joint-#: float
> Default joint angle - specify for each joint in arm (including end effectors the franka arm has 12 joints, 3 of which are fixed)

## scene
*  mode: string `bin-scene`
> The type of scene to load - must be one of the keys in the BaseGymEnv.scene_modes dictionary
*  steps-for-objects-to-fall: int
> Number of times the simulator is stepped after spawning the objects to allow them to fall into place within the bin meant to hold the objects
*  drop-bin-enable: bool
> Whether or not a drop bin is to be loaded into the scene
### bin
*  position: [float, float, float]
> Position of bin
*  orientation: [float, float, float]
> Orientation of bin (in euler and degrees)
#### objects
*  load-specific-object: bool
> Whether to load objects of a random type when the environment is created (when False) or objects of a specific type
*  relative-path-to-object: string
> Relative path from the gym_env/ directory to the .obj file of the object to load (for example: `/assets/training_objs/13x21x5 elektrolit kondenzátor.obj`) OR the name of a simple shape (for enhanced performance). Currently the simple shapes `sphere` and `cylinder` are supported. This option is only used if load-specific-object is True.
*  min-number-of-objects: int
> The min number of objects that may be placed in the bin (the actual number will be randomly selected at runtime and it will be between the minimum and the maximum as specified here), must be less than max-number-of-objects - suggested value is 5
*  max-number-of-objects: int
> The max number of objects that may be placed in the bin (the actual number will be randomly selected at runtime and it will be between the minimum and the maximum as specified here), must be greater than min-number-of-objects - suggested value is 80
### drop-bin
*  position: [float, float, float]
> Position of bin
*  orientation: [float, float, float]
> Orientation of bin (in euler and degrees)

*Note it is inherently assumed that all specified positions and orientations spawn the respective objects in valid locations (i.e. so that the bins are not on top of each other or upside down, etc.)*