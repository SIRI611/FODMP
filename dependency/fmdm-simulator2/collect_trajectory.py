import os
import gym
import time
import random
import pathlib
import gym_env
import numpy as np
from PIL import Image

import os
os.makedirs("datasets", exist_ok=True)

EPISODES = 100
NUM_STEPS = 200
SEED = 112
SAVE_PATH = "datasets/soccer_success_dataset_0.npz"
# CONFIG_DIR = "/home/dalen/your_project/configs/"
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)

def get_ball_ee_state(env, ball_id=None):
    """10D vector: [ball xyz, ball vxyz, EE xyz, gripper_dist].

    Vector representation of the current state of the ball and end effector.

    - ball xyz: world coordinates of the ball
    - ball vxyz: velocity of the ball
    - EE xyz: world coordinates of the end effector
    - gripper_dist: distance between the gripper fingers

    Parameters
    ----------
    env: gym_env.BallIntoGoalEnv
        the environment object
    ball_id: int, optional
        object ID of the ball in the Bullet simulation. If not provided, the
        first target object is used.

    Returns
    -------
    state: np.float32[10]
        the state representation as a numpy array
    """
    if ball_id is None:
        ball_id = env.target_objects[0]['obj_id']  # first target = ball

    (bx, by, bz), _ = env.bullet_client.getBasePositionAndOrientation(ball_id)
    (bvx, bvy, bvz), _ = env.bullet_client.getBaseVelocity(ball_id)

    ee = env.franka_robot.get_end_effector_states()
    ex, ey, ez = ee['position']
    try:
        gripper_dist = env.franka_robot.get_gripper_distance()
    except Exception:
        gripper_dist = 0.0

    return np.array([bx, by, bz, bvx, bvy, bvz, ex, ey, ez, gripper_dist], dtype=np.float32)

if __name__ == '__main__':
    seed_all(SEED)

    # Create env
    env = gym.make('ball_into_goal-v0',
                   config_file=os.environ['path-to-configs'] + 'ball_into_goal.yaml')

    episodes = []
    success_count = 0

    for ep in range(EPISODES):
        _ = env.reset()
        ep_obs, ep_act, ep_rew, ep_done = [], [], [], []
        print(f'[episode {ep+1}/{EPISODES}] reset')

        for t in range(NUM_STEPS):
            # Build compact observation (ball + EE state)
            obs_vec = get_ball_ee_state(env)

            # Use your current expert policy (or swap in your intercept controller)
            action = env.expert_action_one_robot()  # -> [dx,dy,dz,discrete_idx]

            # Step the env
            _, reward, done, info = env.step(action)

            # Log
            ep_obs.append(obs_vec)
            ep_act.append(np.asarray(action, dtype=np.float32))
            ep_rew.append(np.float32(reward))
            ep_done.append(bool(done))

            if done:
                print(f'  finished in {t+1} steps | success={info.get("success", False)}')
                if info.get("success", False):
                    # Keep only successful episodes
                    episodes.append({
                        "observations": np.stack(ep_obs, axis=0),
                        "actions":      np.stack(ep_act, axis=0),
                        "rewards":      np.stack(ep_rew, axis=0),
                        "dones":        np.asarray(ep_done, dtype=bool),
                    })
                    success_count += 1
                break

    # Save only successful episodes
    if episodes:
        np.savez_compressed(SAVE_PATH, episodes=np.array(episodes, dtype=object))
        print(f'\nSaved {success_count} successful episode(s) to {SAVE_PATH}')
    else:
        print('\nNo successful episodes collected — nothing saved.')

    env.close()
