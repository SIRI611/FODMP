## Environment Name: franka-v0

### Attributes
Config file must be structured such that it contains specifications for the following attributes:
*  One Franka arm
*  One bin for picking (initially contains objects)
*  One bin for putting successfully grabbed objects into (initially empty and optional)

### Space Attributes
#### Action space (7 DOF position control + control of both end effectors)
*  A valid action is of the type array<float> * (7 DOF + 2 end effectors) (length 9)
*  Any positive or negative float value can be passed, but it will replaced by the upper or lower bound of the respective joint's range if it is greater than or less than (respectively) the allowed range of values
*  An example is: array([-1.3167408, -1.571633, 1.2385519, -1.7381979, 1.4353446, 0.04333316, 2.1667223, 0.0036893, 0.0335059])

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
