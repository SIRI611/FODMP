# Creating Your Own Robotic Manipulation Gym Environment

## Step 1: Pick Your Robot

Supported robots:
*  Franka Emika
*  Franka Emika Cartesian Robot (pineapple)

This project is structured so that none of the scenes and their related assets (base bin, pick bin, etc.) as well as our basic gym environment for robotic manipulation BaseGymEnv, rely on any specific robot. Refer to the provided gym environments to see if any of the environments that inherit BaseGymEnv support the robot that you desire to simulate. If any of them do, it is better to inherit the basic environment for the desired type of robot (i.e. FrankaBaseEnv for the Franka Emika robot) and build off of that class as opposed to BaseGymEnv. This will provide you with additional functionality that is tailored to the robot of interest. Otherwise inherit BaseGymEnv.

## Step 2: Create Functionality for Your Robot
If the robot that you desire to simulate is not listed above as a supported robot and there is no file corresponding to it in the gym_env/assets/ directory, then it may be worthwhile to model your robot by creating a class (within the gym_env/assets/ directory) that supports the following:
*  Loading the robot into the simulation (via a pybullet connection)
*  Applying configurations to the robot
*  Applying action to the robot (via the desired control method)
*  Getting a current observation of the robot
*  Resetting the robot to default position

Refer to the model (gym_env/assets/franka_arm.py) created for the Franka Emika Robot for guidance.

## Step 3: Implement Abstract Methods
Depending on the class that you inherit, you will have to implement certain abstract methods. In the case that you are inheriting BaseGymEnv you will have to implement the methods and attributes listed below. If you are inheriting a basic environment that is tailored to a robot then the methods and attributes that must be implemented may differ. In this case it is best to either look at either the code for the environment or the relevant section in the documentation (docs/Robotic_Manipulation_Simulator_Doc.pdf) for this project. This documentation also includes important details regarding functionality provided by the included assets.

#### Methods
##### Method `check_termination`
> `def check_termination(self)`

Checks whether or not the simulation has reached a terminating state (method by which this is done depends on
the implementation in a given environment and the config file).
###### Returns
`tuple`
:   A boolean value and a dictionary. The boolean value represents whether or not the simulation is in a
    terminating state and the dictionary contains auxillary diagnostic info (useful for debugging and sometimes
    training).

##### Method `get_reward`
> `def get_reward(self)`

Calculates the reward based on the current state of the agent and the environment.
###### Returns
`float`
:   Value of the reward.

##### Method `get_state`
> `def get_state(self)`

Gets the current state of the robots(s) in the environment.
###### Returns
`object`
:   Observation of the current state of the robot(s) in the environment

##### Method `reset`    
> `def reset(self, keep_object_type=False)`

Resets class-relevant attributes (i.e. robots should be reset to their starting position).

##### Method `set_space_attributes`
> `def set_space_attributes(self)`

Defines the action space, observation space, and reward range in terms of the types provided by python package
gym.spaces (i.e. spaces.Dict(), spaces.Box()) and sets them as instance attributes.
    
##### Method `step`
> `def step(self, action)`

Runs one step of the simulation.

#### Attributes
##### Attribute `observation_space`
The observation space in your environment.

##### Attribute `action_space`
The action space in your environment.

##### Attribute `reward_range`
The reward range in your environment.

## Step 4: Register Your Environment
1.  Ensure the python source code file for your environment is located in the directory gym_env/envs/
2.  Ensure the class representing your gym environment is imported in the gym_env/envs/__init__.py file in the form 
> `from gym_env.envs.[file name] import [class name]`
3.  Register your environment in gym_env/__init__.py in the form `register(id=[environment name], entry_point='gym_env.envs:[class name])` where environment name follows the regular expression [a-zA-Z]+-v[0-9]+

## Additional Information

At this point all of the neccessary methods and attributes should be implemented so feel free to add any additional functionality that you feel may improve the quality of the simulation. See below for some additional notes:

*  The scene that is simulated is specified in the configuration file. For more information on how this is done, see the Scene section on the Configuration File wiki page

*  It is a good idea to use `assert` in the __init__ portion of the class representing your environment to ensure that the configuration file is structured correctly  (i.e. ensure that there are only specifications for one arm in the configuration file if you want your environment to only support one arm).

*  For guidance, refer to the provided gym environments