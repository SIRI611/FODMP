import pytest
import pybullet as pb
import gym_env.


@pytest.fixture(scope='module')
def pb_client():
    client = pb.connect(pb.DIRECT)
    yield client
    pb.disconnect(client)
