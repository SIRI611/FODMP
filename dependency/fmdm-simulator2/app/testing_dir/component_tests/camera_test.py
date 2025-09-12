import pytest
import pybullet as pb
import numpy as np


@pytest.mark.component
class TestCamera:
    def test_init(pb_client, camera_config, camera_instance):
        height_res = camera_config['projection-matrix']['height-resolution']
        width_res = camera_config['projection-matrix']['width-resolution']
        correct_view_matrix = pb_client.computeViewMatrix(
            cameraEyePosition=camera_config['view-matrix']['eye-position'],
            cameraTargetPosition=camera_config['view-matrix']['target-position'],
            cameraUpVector=camera_config['view-matrix']['up-vector']
        )
        correct_projection_matrix = pb_client.computeProjectionMatrixFOV(
            fov=camera_config['projection-matrix']['fov'],
            aspect=width_res / height_res,
            nearVal=camera_config['projection-matrix']['near-plane'],
            farVal=camera_config['projection-matrix']['far-plane']
        )
        assert camera_instance.view_matrix == correct_view_matrix
        assert camera_instance.projection_matrix == correct_projection_matrix

    def test_adjust_view_matrix(pb_client, camera_config, camera_instance):
        random_eye_pos = np.random.uniform(size=(3,))
        random_target_pos = np.random.uniform(size=(3,))
        random_up_vector = np.random.randint(2, size=(3,))
        camera_instance.adjust_view_matrix(random_eye_pos, random_target_pos, random_up_vector)
        correct_adj_view_matrix = pb_client.computeViewMatrix(
            cameraEyePosition=random_eye_pos,
            cameraTargetPosition=random_target_pos,
            cameraUpVector=random_up_vector
        )
        assert camera_instance.view_matrix == correct_adj_view_matrix

    def test_get_image(pb_client, camera_config, testing_sim_mode, camera_instance):
        view_matrix = pb_client.computeViewMatrix(
            cameraEyePosition=camera_config['view-matrix']['eye-position'],
            cameraTargetPosition=camera_config['view-matrix']['target-position'],
            cameraUpVector=camera_config['view-matrix']['up-vector']
        )
        projection_matrix = pb_client.computeProjectionMatrixFOV(
            fov=camera_config['projection-matrix']['fov'],
            aspect=width_res / height_res,
            nearVal=camera_config['projection-matrix']['near-plane'],
            farVal=camera_config['projection-matrix']['far-plane']
        )
        correct_image_scale = 255
        human = camera_instance.get_image(testing_sim_mode, 'human')
        assert human == None
        rgb_array = camera_instance.get_image(testing_sim_mode, 'rgb_array')
        _, _, rgb_px, _, _ = pb_client.getCameraImage(
            width=camera_config['width-resolution'],
            height=camera_config['height-resolution'],
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=pb_client.ER_TINY_RENDERER
        )
        correct_rgb_array = np.array(rgb_px)[:, :, :3]
        assert (rgb_array == correct_rgb_array).all()
        _, _, rgb_px, depth_px, _ = pb_client.getCameraImage(
            width=camera_config['width-resolution'],
            height=camera_config['height-resolution'],
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=pb_client.ER_TINY_RENDERER
        )
        correct_rgb_array = np.array(rgb_px)[:, :, :3]
        camera_instance.scale_depth_data = False
        rgb_array, depth_array = camera_instance.get_image(testing_sim_mode, 'rgb_and_depth_arrays')
        correct_depth_array = camera_config['far-plane'] * camera_config['near-plane'] / (camera_config['far-plane'] \
            - ((camera_config['far-plane'] - camera_config['far-plane']) * np.array(depth_px)))
        assert (rgb_array == correct_rgb_array).all()
        assert (depth_array == correct_depth_array).all()
        camera_instance.scale_depth_data = True
        rgb_array, depth_array = camera_instance.get_image(testing_sim_mode, 'rgb_and_depth_arrays')
        correct_scaled_depth_array = (correct_depth_array * correct_image_scale).astype(np.uint8)
        assert (rgb_array == correct_rgb_array).all()
        assert (depth_array == correct_scaled_depth_array).all()

    def test_render(pb_client, camera_config, testing_sim_mode, camera_instance):
        num_colors = 3
        human = camera_instance.render(testing_sim_mode, 'human')
        assert human == None
        rgb_array = camera_instance.render(testing_sim_mode, 'rgb_array')
        assert rgb_array.shape == (camera_config['height-resolution'], camera_config['width-resolution'], num_colors)
        rgb_array, depth_array = camera_instance.render(testing_sim_mode, 'rgb_and_depth_arrays')
        assert rgb_array.shape == (camera_config['height-resolution'], camera_config['width-resolution'], num_colors)
        assert depth_array.shape == (camera_config['height-resolution'], camera_config['width-resolution'])
