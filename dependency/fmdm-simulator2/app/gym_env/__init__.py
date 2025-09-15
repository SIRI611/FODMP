import os
import pathlib
import gym_env.components
import gym_env.envs
import gym_env.utils
from gym.envs.registration import register
import pybullet_data


register(
    id='franka-v0',
    entry_point='gym_env.envs:FrankaEnv0',
)

register(
   id='franka-v1',
   entry_point='gym_env.envs:FrankaEnv1',
)

register(
   id='franka-v2',
   entry_point='gym_env.envs:FrankaEnv2',
)

register(
   id='cartesian-v0',
   entry_point='gym_env.envs:CartesianEnv0',
)

register(
   id='placement-v1',
   entry_point='gym_env.envs:PlacementEnv0'
)

register(
   id='placement-v4',
   entry_point='gym_env.envs.placement_v4:PlacementEnv4'
)

register(
   id='collect_clear-v0',
   entry_point='gym_env.envs.collect_and_clear:CollectAndClearEnv'
)

register(
   id='put_donut-v0',
   entry_point='gym_env.envs.put_donut:PutDonutEnv'
)

register(
   id='put_mug-v0',
   entry_point='gym_env.envs.put_mug:PutMugEnv'
)

register(
   id='place_cup-v0',
   entry_point='gym_env.envs.place_cup:PlaceCupEnv'
)


register(
   id='place_hammer-v0',
   entry_point='gym_env.envs.place_hammer:PlaceHammerEnv'
)

register(
   id='ball_into_goal-v0',
   entry_point='gym_env.envs.ball_into_goal:BallIntoGoalEnv'
)
# register(
#     id='pineapple-diff-v0',
#     entry_point='gym_env.envs:PineappleDiffCartEnv',
# )
#
# register(
#     id='pineapple-calib-v0',
#     entry_point='gym_env.envs:PineappleCalibratedEnv',
# )

os.environ['path-to-configs'] = str(pathlib.Path(__file__).parent.parent / 'configs') + '/'
os.environ['path-to-gym-env'] = str(pathlib.Path(__file__).parent) + '/'
os.environ['path-to-assets'] = str(pathlib.Path(__file__).parent / 'assets') + '/'
os.environ['path-to-bins'] = str(pathlib.Path(__file__).parent / 'assets' / 'bins') + '/'
os.environ['path-to-table'] = str(pathlib.Path(__file__).parent / 'assets' / 'table') + '/'
os.environ['path-to-robots'] = str(pathlib.Path(__file__).parent / 'assets' / 'robots') + '/'
os.environ['path-to-testing-objs'] = str(pathlib.Path(__file__).parent / 'assets' / 'testing_objs') + '/'
os.environ['path-to-training-objs'] = str(pathlib.Path(__file__).parent / 'assets' / 'training_objs') + '/'
os.environ['path-to-slots'] = str(pathlib.Path(__file__).parent / 'assets' / 'slots') + '/'
os.environ['path-to-pegs'] = str(pathlib.Path(__file__).parent / 'assets' / 'pegs') + '/'
os.environ['path-to-new-scenes'] = str(pathlib.Path(__file__).parent / 'assets' / 'new_scenes') + '/'
