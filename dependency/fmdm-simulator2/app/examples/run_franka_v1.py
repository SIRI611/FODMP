import os
import gym
import time
import gym_env

SIM_STEPS = 50


def run():
    """
    Steps the custom franka gym environment by applying actions to the franka arm in the environment until a terminating
    condition is met. Then returns the time that it took for the franka arm to reach a terminating state.

    Returns
    -------
    float
        The time that it took for the franka arm to reach a terminating state.
    """
    done = False
    # Gets the start time
    s = time.monotonic()
    f = time.monotonic()
    # Creates franka gym environment
    env = gym.make('franka-v1', config_file=os.environ['path-to-configs'] + 'franka1.yaml')
    observation = env.reset()
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(info)
        # Physics engine is stepped a certain number of times so that the changes in velocity result in noticable
        # changes in the robots position
        for _ in range(SIM_STEPS):
            env.bullet_client.stepSimulation()
    env.close()
    f = time.monotonic()
    return f - s


if __name__ == '__main__':
    """
    Runs the example.
    """
    print('Time =', run())
