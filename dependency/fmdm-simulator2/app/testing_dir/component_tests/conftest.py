import pathlib
import sys
import pytest
import pathlib
import yaml
import gym_env.scenes

# pytest_plugin = 'testing_dir.helpers.helper_fixtures'


@pytest.fixture(scope='session')
def camera_config():
    cwd_path = pathlib.Path(__file__).parent
    conf_file_path = str(cwd_path) + '/component_test_config.yaml'
    conf_file = open(conf_file_path)
    component_test_config = yaml.safe_load(conf_file)
    return component_test_config['camera']


@pytest.fixture(scope='module')
def camera_instance(camera_config, pb_client):
    return camera.Camera(camera_config, pb_client)


@pytest.fixture(scope='session')
def testing_sim_mode():
    component_test_config = yaml.safe_load('component_test_config.yaml')
    return component_test_config['simulation']['sim-mode']
