import pytest

@pytest.fixture(scope=function)
def load_config_into_dict(path_to_config_file):
    """
    Opens and reads the config file at the location given by the param path_to_config_file and returns the data in
    dictionary form.

    Parameters
    ----------
    path_to_config_file : str
        Path to yaml file consisting of simulation configuration.
    Returns
    -------
    dict
        The config file data as a dictionary.
    """
    simulator_config = {}
    with open(config_file) as config:
        try:
            simulator_config = yaml.safe_load(config)
        except Exception as err:
            print('Error Configuration File:{}'.format(err))
            raise err
    return simulator_config