# Robotic Manipulation Simulator
The Robotic Manipulation Simulator is a toolkit aimed at facilitating the efficient development of machine learning
algorithms. The long term goal is to provide researchers with the support and configurability to train multiple types of
robots on different peg-in-hole / bin picking scenarios, in a highly configurable and easy-to-use environment that is capable of a
high degree of randomization.   

![grab_example2](docs/example_pics/insertion_example.PNG)

TODO: complete the documentation.

For bin-picking example: README_Backup.md

# Installation

### Source Method
> Navigate into robotic-manipulation project app directory

`cd <project>/app/`

> Install all minimum project requirements

`pip install -r requirements.txt` or `pip3 install -r requirements.txt` depending on your setup
 
> Install python package and dependencies for project

`pip install -e .` or `pip3 install -e .` depending on your setup

### Usage

> run a standard Gym env

`python examples/insertion_test.py`

> Step by step instructions

```
env = gym.make('placement-v1', config_file='path_to_scene_config_file.yaml')
max_joint_velocities = [2.175, 2.175, 2.175, 2.175, 2.61, 2.61]  # 2.61
EPISODES = 1000
NUM_STEPS = 200
for m in range(EPISODES):
    observation = env.reset()
    for i in range(NUM_STEPS):
        action = random_joint_vel(max_joint_velocities)
        observation, reward, done, info = env.step(action)
env.close()
```

A function generates random actions:
```
def random_joint_vel(max_joint_vel):
    scale = 0.1
    return [random.uniform(-max_vel*scale, max_vel*scale) for max_vel in max_joint_vel]
```