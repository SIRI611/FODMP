import os
import pytest
import yaml
from gym_env.envs.franka_v0 import Franka0Env
from gym_env.envs.franka_v1 import Franka1Env


@pytest.fixture(scope='session')
def path_to_testing_dir():
    return os.getcwd()



@pytest.fixture(scope='session')
def env_instance(pytestconfig):
    """
    Session-scoped fixture that returns an instance of a custom gym env object (the specific env that is loaded depends
    on the markers that are used. This function checks the markers that are passed and ensures that they are a valid
    combination. It then returns the specified env or prints an error message. The default env that is returned (if none
    are specified) is an instance of Franka0Env or if the marker "not franka_v0" is used then an instance of Franka1Env
    will be used and so on (in numerical order).

    Parameters
    ----------
    pytestconfig : object
        Session-scoped fixture that returns the _pytest.config.Config object (as per the pytest official docs - see them
        for more info).
    """
    # TODO add any new envs here
    env_dict = {'franka_v0': Franka0Env, 'franka_v1': Franka1Env}
    not_allowed_envs = []
    markers = pytestconfig.getoption('-m')
    count, version, env_name, env_to_load = 0, 0, 'franka_v', None
    for env in env_dict.keys():
        if (env in markers) and ('not ' + env not in markers):
            count += 1
            env_to_load = env
        elif 'not ' + env in markers:
            not_allowed_envs.append(env)
    assert count in [0, 1], 'Invalid marker combination. Please specify only one or no (default) specific env marker ' \
                            'at a time.'
    assert len(env_dict) > len(not_allowed_envs), 'At least one env must be allowed.'
    if count == 1:
        env_instance = env_dict[env_to_load](config_file=os.environ['env_path'] + '/configs/test_config.yaml')
        env_instance.reset()
        yield env_instance
    else:
        while env_name + str(version) in not_allowed_envs:
            version += 1
        env_instance = env_dict[env_name + str(version)](config_file=os.environ['env_path'] + '/configs/test_config.yaml')
        env_instance.reset()
        yield env_instance
    env_instance.close()
    assert env_instance.physics_client_id == -1, 'BaseGymEnv.close() did not disconnect from pybullet and dump all ' \
                                                 'services correctly.'


@pytest.fixture(scope='session')
def config():
    """
    Session-scoped fixture that returns the config file as a dictionary.

    Returns
    -------
    dict
    """
    config_file_path = os.environ['env_path'] + '/configs/test_config.yaml'
    with open(config_file_path) as config:
        try:
            return yaml.safe_load(config)
        except Exception as err:
            print('Error Configuration File:{}'.format(err))
        raise err


@pytest.fixture(scope='session')
def num_franka_joints():
    """
    Session-scoped fixture that returns the fixed number of joints in Franka arm (as specified by the urdf we are
    using).

    Returns
    -------
    int
    """
    return 12
