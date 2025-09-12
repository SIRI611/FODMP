import os
import pytest
import random
import numpy as np


@pytest.mark.base_functionality
class TestPickBin:
    def test_load_objects(self, env_instance, config):
        """
        Tests the load_objects method of BinPicking. Ensures that all objects have been successfully loaded into
        the simulation by checking that all their ids are greater than -1. Also tests the keep_object_type option.
    
        Parameters
        ----------
        env_instance : gym_env.envs.*
            An instance of a gym env.
        config : dict
            File consisting of simulation config.
        """
        original_obj_type = env_instance.scene.bin_with_objs.path_to_current_object
        env_instance.reset(keep_object_type=True)
        new_obj_type = env_instance.scene.bin_with_objs.path_to_current_object
        objects_are_valid = np.array([obj_id > -1 for obj_id in env_instance.scene.bin_with_objs.objects])
        assert objects_are_valid.all()
        assert original_obj_type == new_obj_type
        if config['scene']['bin']['objects']['load-specific-object']:
            specific_obj_type = os.environ['env_path'] + config['scene']['bin']['objects']['relative-path-to-object']
            original_obj_type = env_instance.scene.bin_with_objs.path_to_current_object
            assert specific_obj_type == original_obj_type
            env_instance.reset()
            new_obj_type = env_instance.scene.bin_with_objs.path_to_current_object
            assert original_obj_type == new_obj_type

    def test_get_random_object_file(self, env_instance, config):
        """
        Tests the get_random_object_file method by making sure different object files are being loaded according
        to the ml-mode in the config file. It goes through at most 10 runs as the chance that the same object file
        is randomly picked 10 times in a row (out of > 300 possible files) is practically impossible, thus it can
        be concluded that if the same file is picked 10 times in a row then something is wrong.
        
        Parameters
        ----------
        env_instance : gym_env.envs.*
            An instance of a gym env.
        config : dict
            File consisting of simulation config.
        """
        if not config['scene']['bin']['objects']['load-specific-object']:
            trials = 10
            ml_mode = 'training_objs' if config['simulation']['ml-mode'] == 'training' else 'testing_objs'
            different_object_file_loaded = False
            for _ in range(trials):
                original_obj_type = env_instance.scene.bin_with_objs.path_to_current_object
                env_instance.reset(keep_object_type=False)
                new_obj_type = env_instance.scene.bin_with_objs.path_to_current_object
                assert (ml_mode in original_obj_type) and (ml_mode in new_obj_type)
                if original_obj_type != new_obj_type:
                    different_object_file_loaded = True
                    break
            assert different_object_file_loaded

    def test_get_list_object_coords(self, env_instance, config):
        """
        Tests the get_lit_objet_coords method of BinPicking. Ensures that the correct number of positions are
        generated and that they are all in a valid location.
        
        Parameters
        ----------
        env_instance : gym_env.envs.*
            An instance of a gym env.
        config : dict
            File consisting of simulation config.
        """
        min_objects = config['scene']['bin']['objects']['min-number-of-objects']
        max_objects = config['scene']['bin']['objects']['max-number-of-objects']
        number_of_objs = random.randint(min_objects, max_objects)
        sample_list = env_instance.scene.bin_with_objs.get_list_object_coords(number_of_objs)
        assert len(sample_list) == number_of_objs
        for obj_position in sample_list:
            assert in_vicinity_of_bin(env_instance, obj_position, ignore_z=True)

    def test_contains(self, env_instance, config):
        """
        Tests the contains method of BinPicking. Ensures the method is performing as expected.
        
        Parameters                                              
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        """
        min_objects = config['scene']['bin']['objects']['min-number-of-objects']
        max_objects = config['scene']['bin']['objects']['max-number-of-objects']
        number_of_objs = random.randint(min_objects, max_objects)
        sample_list = env_instance.scene.bin_with_objs.get_list_object_coords(number_of_objs)
        assert np.array([env_instance.scene.bin_with_objs.contains(obj_pos) == 
                         in_vicinity_of_bin(env_instance, obj_pos) for obj_pos in sample_list]).all()

    def test_remove_single_object_from_bin(self, env_instance):           
        """                                                          
        Tests the remove_single_object method of BinPicking. Ensures that an object is removed from the
        pybullet simulation and the change in number of objects is properly tracked by BinScene and BinPicking.                                        
        
        Parameters                                              
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        """
        initial_total_number_of_bodies = env_instance.bullet_client.getNumBodies()
        initial_number_of_objects = len(env_instance.scene.bin_with_objs.objects)
        random_object = random.choice(env_instance.scene.bin_with_objs.objects)
        env_instance.remove_single_object_from_bin(random_object)
        final_total_number_of_bodies = env_instance.bullet_client.getNumBodies()
        final_number_of_objects = len(env_instance.scene.bin_with_objs.objects)
        assert final_total_number_of_bodies == initial_total_number_of_bodies - 1
        assert final_number_of_objects == initial_number_of_objects - 1
        assert random_object not in env_instance.scene.bin_with_objs.objects

    def test_remove_objs_not_in_bin(self, env_instance):
        """
        Tests the remove_objs_not_in_bin method of BinPicking. Calls the method and then ensures that all remaining
        objects are in the bin.

        Parameters
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        """
        initial_total_number_of_bodies = env_instance.bullet_client.getNumBodies()
        initial_number_of_objects = len(env_instance.scene.bin_with_objs.objects)
        env_instance.remove_objects_not_in_bins()
        position = 0
        for obj in env_instance.scene.bin_with_objs.objects:
            obj_position = env_instance.bullet_client.getBasePositionAndOrientation(obj)[position]
            assert in_vicinity_of_bin(env_instance, obj_position)
        final_total_number_of_bodies = env_instance.bullet_client.getNumBodies()
        final_number_of_objects = len(env_instance.scene.bin_with_objs.objects)
        assert (final_total_number_of_bodies - initial_total_number_of_bodies == final_number_of_objects -
                initial_number_of_objects)
                
    def test_reset(self, env_instance):
        """
        Tests the reset method of BinPicking. Ensures objects are being reloaded and the keep_object_type option
        works.
        
        Parameters
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        """
        assert len(env_instance.scene.bin_with_objs.objects) > 0
        original_obj_type = env_instance.scene.bin_with_objs.path_to_current_object
        env_instance.reset(keep_object_type=True)
        new_obj_type = env_instance.scene.bin_with_objs.path_to_current_object
        assert original_obj_type == new_obj_type


# HELPER FUNCTIONS
def in_vicinity_of_bin(env_instance, obj_position, ignore_z=False):
    """
    Checks if a given object position is within the vicinty of the bin.
    
    Parameters
    ----------
    env_instance : gym_env.envs.franka_v#.Franka#Env
        An instance of a franka gym env.
    obj_position : list
        Object position, in the form [x, y, z]
    ignore_z : bool
        Whether or not to factor the z position of the object into consideration
    """
    x, y, z = 0, 1, 2
    length, width, height = env_instance.scene.bin_with_objs.dimensions
    min_x, max_x = env_instance.scene.bin_with_objs.bin_origin[x] - (length / 2), \
                   env_instance.scene.bin_with_objs.bin_origin[x] + (length / 2)
    min_y, max_y = env_instance.scene.bin_with_objs.bin_origin[y] - (width / 2), \
                   env_instance.scene.bin_with_objs.bin_origin[y] + (width / 2)
    if ignore_z:
        return (min_x <= obj_position[x] <= max_x) and (min_y <= obj_position[y] <= max_y)
    min_z, max_z = env_instance.scene.bin_with_objs.bin_origin[z], \
                   env_instance.scene.bin_with_objs.bin_origin[z] + (height * 2)
    return (min_x <= obj_position[x] <= max_x) and (min_y <= obj_position[y] <= max_y) and \
           (min_z <= obj_position[z] <= max_z)
