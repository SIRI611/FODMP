import os
import gym
import time
import random
import pathlib
import gym_env
import numpy as np
import pickle

EPISODES = 100000
NUM_STEPS = 100
SEED = 112  # random.randint(0,99999999999)
SAVE_PICKLE_FILE_EVERY_NUM_EPS = 50
sample_folder = '/home/jun/gato_data/downstream_task/place_cup/'
# sample_folder = 'samples/'
def wrap_data_dict(expert_samples):
    data = {}
    vect_observations = []
    actions = []
    rewards = []
    terminals = []
    images = []

    # mujoco_physics_engine_states = []
    for i in range(len(expert_samples)):
        vect_observations.append(expert_samples[i][0]['state'])
        actions.append(expert_samples[i][1])
        rewards.append(expert_samples[i][2])
        terminals.append(expert_samples[i][4])
        image_data = np.array([expert_samples[i][0]['top_view_img'],expert_samples[i][0]['corner1_img'],expert_samples[i][0]['corner2_img']])
        images.append(image_data)
        # mujoco_physics_engine_states.append(expert_samples[i][6])

    data['observations'] = np.array(vect_observations)
    data['actions'] = np.array(actions)
    data['rewards'] = np.array(rewards)
    data['terminals'] = np.array(terminals)
    data['images'] = np.array(images)
    # data['mujoco_physics_engine_states'] = np.array(mujoco_physics_engine_states)
    return data

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)

# 0.707, -0.707, 0, 0

if __name__ == '__main__':
    # Gets the start time
    s = time.monotonic()
    f = time.monotonic()
    # seed_all(SEED)
    # Creates franka gym environment
    env = gym.make('place_cup-v0', config_file=os.environ['path-to-configs'] + 'place_cup.yaml')
    sample_collection_buffer = []
    eps_collected = 0
    for m in range(EPISODES):
        obs = env.reset()
        traj_buffer = []
        for i in range(NUM_STEPS):
            step_s = time.monotonic()  # start time of the step
            action = env.expert_action_one_robot()
            # print('==put donot eps collected {:3d} {:4d}/{}, step {:4d}/{}, action: ['.format(eps_collected, m+1, EPISODES, i, NUM_STEPS), end='')
            # for k in action:
            #     print('{:+.2f} '.format(k), end='')
            next_obs, reward, done, info = env.step(action)
            done_no_max = done and i < NUM_STEPS - 1
            sample_array = [obs, action, reward, next_obs, done, done_no_max]
            traj_buffer.append(sample_array)
            obs = next_obs
            # print('] Duration: {:2.3f}s, Reward: {:+.2f}  Total time spent: {:4.1f}]'.format(
            #     time.monotonic() - step_s,reward, time.monotonic() - s))
            print('==place Cup eps collected {:3d} {:4d}/{}, Duration: {:2.3f}s, Total time spent: {:4.1f}, step {}/{}'.format(
                eps_collected, m + 1,
                EPISODES, time.monotonic() - step_s, time.monotonic() - s, i, NUM_STEPS))
            if done and i+1>30:
                print('==place Cup eps finished in {} steps'.format(i+1))
                eps_collected += 1
                sample_collection_buffer.append(traj_buffer)
                break

        if (m+1) % SAVE_PICKLE_FILE_EVERY_NUM_EPS == 0:
            # prepare pickle file
            data_collection_list = []
            # expert_samples = np.load('{}/{}/{}'.format(log_dir, env_id, "samples.npy"), allow_pickle=True)
            for traj_data in sample_collection_buffer:
                data_collection_list.append(wrap_data_dict(traj_data))
            sample_collection_buffer=[]
            rand_pickle_file_name = '{}.pkl'.format(random.randint(0,99999999999))
            with open(sample_folder+rand_pickle_file_name, 'wb') as f:
                pickle.dump(data_collection_list, f)
            print('saved to file {}'.format(m+1))
    env.close()
    f = time.monotonic()
#
# import ssl
#
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     # Legacy Python that doesn't verify HTTPS certificates by default
#     pass
# else:
#     # Handle target environment that doesn't support HTTPS verification
#     ssl._create_default_https_context = _create_unverified_https_context
