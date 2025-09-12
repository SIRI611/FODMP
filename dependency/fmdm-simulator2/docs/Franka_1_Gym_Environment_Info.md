## Environment Name: franka-v1

### Attributes
Config file must be structured such that it contains specifications for the following attributes:
*  One Franka arm
*  One bin for picking (initially contains objects)
*  One bin for putting successfully grabbed objects into (initially empty and optional)

### Space Attributes
#### Action space (7 DOF velocity control + control of both end effectors)
*  A valid action is of the type (7 DOF + 2 end effectors) (length 9)
*  Any positive or negative float value can be passed, but it will replaced by the upper or lower bound of the respective joint's range if it is greater than or less than (respectively) the allowed range of values
*  An example is: array([-43.682396, -47.26348, 56.55585, 89.566956, -2.6881008, 16.764774, -13.375332, 83.66662, -41.79764])

#### Observation space
*  The observation space is an tuple consisting of an array of the rgb camera render data and an array of the depth camera render data
    *  The rgb array has the shape (height_resolution, width_resolution, 3) with height and width resolution being specified in the configuration file
    *  The rgb array has the shape (height_resolution, width_resolution) with height and width resolution being specified in the configuration file

### Reward Calculation
*  The reward calculation happens as follows:
    *  If the robot has collided with the bin with objects, a reward of -1 is given
    *  If the robot executes a successful grab, a reward of +1 is given
    *  Otherwise a reward of -0.1 is given

### Terminating Conditions
The simulator will terminate if one of the following takes place (possible to enable/disable the termination conditions in the configuration file):
*  The Franka arm has collided with the bin with objects
*  The Franka arm has exceeded the number of successful grabs before termination