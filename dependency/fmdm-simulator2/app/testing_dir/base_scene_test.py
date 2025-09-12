import pytest


@pytest.mark.base_functionality
class TestBaseScene:
    def test_reset(self, env_instance, config):
        """
        Tests the reset method of BaseScene. Ensures that all the check-able physics engine parameters that are set are set
        correctly.

        Parameters
        ----------
        env_instance : gym_env.envs.*
            An instance of a gym env.
        config : dict
            File consisting of simulation config.
        """
        current_physics_engine_params = env_instance.bullet_client.getPhysicsEngineParameters()
        correct_fixed_time_step = config['simulation']['timestep']
        correct_num_sub_steps = config['simulation']['number-of-sub-steps']
        correct_num_solver_iterations = config['simulation']['number-of-solver-iterations']
        correct_gravity = [0, 0, -config['simulation']['gravity']]
        assert current_physics_engine_params['fixedTimeStep'] == correct_fixed_time_step
        assert current_physics_engine_params['numSubSteps'] == correct_num_sub_steps
        assert current_physics_engine_params['numSolverIterations'] == correct_num_solver_iterations
        assert current_physics_engine_params['gravityAccelerationX'] == correct_gravity[0]
        assert current_physics_engine_params['gravityAccelerationY'] == correct_gravity[1]
        assert current_physics_engine_params['gravityAccelerationZ'] == correct_gravity[2]
        original_obj_type = env_instance.scene.bin_with_objs.path_to_current_object
        env_instance.reset(keep_object_type=True)
        new_obj_type = env_instance.scene.bin_with_objs.path_to_current_object
        assert original_obj_type == new_obj_type
