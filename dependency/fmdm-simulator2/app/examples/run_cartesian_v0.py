import os
import gym
import time
import gym_env # to call __init__.py
# Gets the start time
s = time.monotonic()
# Creates gym environment for cartesian robot
env = gym.make('cartesian-v0', config_file=os.environ['path-to-configs'] + 'cartesian0.yaml')
# logId = env.bullet_client.startStateLogging(env.bullet_client.STATE_LOGGING_PROFILE_TIMINGS, "timings.json")
# obs = env.reset()
num_rollouts = 100
num_steps = 10000

for m in range(num_rollouts):
    observation = env.reset()
    for i in range(num_steps):
        step_s = time.monotonic()  # start time of the step
        action = env.sample_actions()
        pos, yaw, d = action[0:3], action[3], action[4]
        print('[x:{:+.4f}, y:{:+.4f}, z:{:+.4f}, phi:{:+.2f}, d:{:+.4f}]'.format(pos[0], pos[1], pos[2], yaw, d), end='')
        observation, reward, done, info = env.step(action)
        if done:
            if info["collide-bin"]:  # colliding with bin walls
                print('\n      Result: [Colliding with bin walls!] ')
            if info["success-pick"]:
                print('\n      Result: [==successful== grasp obj id {:3d}] '.format(info["obj-id"]))
                # do sth here. like reset env.
        else:
            print('\n      Result: [nothing learned] ')

        print('      Duration: {:2.3f}s  Completion : Total time spent [{:4d}/{:4d}, {:4.1f}]'.format(
          time.monotonic() - step_s, info["grasped-obj-num"], info["total-obj-num"], time.monotonic() - s))
