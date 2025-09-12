import math
import pytest
import random
import numpy as np
from gym_env.components import *


@pytest.mark.base_functionality
class TestBinScene:
    def test_reset(self, env_instance, config):
        """
        Tests the reset method of BinScene. Ensures that the ground has been successfully loaded as well as the bin
        for bin picking and that the current_number_of_objects attribute is tracked correctly. Also ensures that
        the keep_object_type option works.
        
        Parameters
        ----------
        env_instance : gym_env.envs.*
            An instance of a gym env.
        config : dict
            File consisting of simulation config.
        """
        assert env_instance.scene.ground_id > -1
        assert (isinstance(env_instance.scene.drop_bin, base_bin.BaseBin) if config['scene']['drop-bin-enable'] else
                env_instance.scene.drop_bin == None)
        assert isinstance(env_instance.scene.bin_with_objs, pick_bin.PickBin)
        assert env_instance.scene.bin_with_objs.steps_for_objects_to_fall == config['scene']['steps-for-objects-to-fall']
        assert env_instance.scene.current_number_of_objects == len(env_instance.scene.bin_with_objs.objects)
        original_obj_type = env_instance.scene.bin_with_objs.path_to_current_object
        env_instance.reset(keep_object_type=True)
        new_obj_type = env_instance.scene.bin_with_objs.path_to_current_object
        assert original_obj_type == new_obj_type

    def test_load_ground(self, env_instance):
        """
        Tests the load_ground method of BinScene. Ensures that the ground has been successfully loaded into the
        simulation by making sure the unique id assigned to the ground is greater than -1.
        
        Parameters
        ----------
        env_instance : gym_env.envs.*
            An instance of a gym env.
        """
        assert env_instance.scene.ground_id > -1

    def test_load_bin(self, env_instance, config):
        """
        Tests the load_bin method of BinScene. Ensures that all the bin and drop bin (depending on config) have been
        loaded into the simulation successfully and are instances of the correct class.
    

        Parameters
        ----------
        env_instance : gym_env.envs.*
            An instance of a gym env.
        config : dict
            File consisting of simulation config.
        """
        assert (isinstance(env_instance.scene.drop_bin, base_bin.BaseBin) if config['scene']['drop-bin-enable'] else
                env_instance.scene.drop_bin == None)
        assert isinstance(env_instance.scene.bin_with_objs, pick_bin.PickBin)
        assert env_instance.scene.bin_with_objs.steps_for_objects_to_fall == config['scene']['steps-for-objects-to-fall']
        assert env_instance.scene.current_number_of_objects == len(env_instance.scene.bin_with_objs.objects)
        
    def test_remove_objects_not_in_bins(self, env_instance):
        """
        Tests the remove_objects_not_in_bins method of FrankaBaseEnv. Calls the method and then ensures that all remaining
        objects are in the bin.

        Parameters
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        """
        initial_total_number_of_bodies = env_instance.bullet_client.getNumBodies()
        initial_number_of_objects = env_instance.scene.current_number_of_objects
        env_instance.scene.remove_objects_not_in_bins()
        position = 0
        for obj in env_instance.scene.bin_with_objs.objects:
            obj_position = env_instance.bullet_client.getBasePositionAndOrientation(obj)[position]
            assert env_instance.scene.bin_with_objs.contains(obj_position)
        final_total_number_of_bodies = env_instance.bullet_client.getNumBodies()
        final_number_of_objects = env_instance.scene.current_number_of_objects
        assert (final_total_number_of_bodies - initial_total_number_of_bodies == final_number_of_objects -
                initial_number_of_objects)

    def test_remove_single_object_from_bin(self, env_instance):           
        """                                                          
        Tests the remove_single_object_from_bin method of BinScene. Ensures that an object is removed from the
        pybullet simulation and the change in number of objects is properly tracked.
        
        Parameters                                              
        ----------
        env_instance : gym_env.envs.franka_v#.Franka#Env
            An instance of a franka gym env.
        """
        initial_total_number_of_bodies = env_instance.bullet_client.getNumBodies()
        initial_number_of_objects = env_instance.scene.current_number_of_objects
        random_object = random.choice(env_instance.scene.bin_with_objs.objects)
        env_instance.scene.remove_single_object_from_bin(random_object)
        final_total_number_of_bodies = env_instance.bullet_client.getNumBodies()
        final_number_of_objects = env_instance.scene.current_number_of_objects
        assert final_total_number_of_bodies == initial_total_number_of_bodies - 1
        assert final_number_of_objects == initial_number_of_objects - 1
        assert random_object not in env_instance.scene.bin_with_objs.objects
