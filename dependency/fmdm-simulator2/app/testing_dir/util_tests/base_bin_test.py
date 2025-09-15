import math
import pytest
import pybullet as pb
from gym_env.components.base_bin import BaseBin

@pytest.mark.bin_functionality
class TestBaseBin:
    default_dimensions = [0.16, 0.16, 0.04]
    default_mesh_scale = [1, 1, 1]

    def __init__(self, base_bin_config):
        self.pb_instance = pb.connect(pb.DIRECT)
        self.correct_bin_origin = base_bin_config['position']
        self.correct_bin_orientation = pb.getQuaternionFromEuler(
            [math.radians(angle) for angle in base_bin_config['orientation']]
        )
        if 'dim' in base_bin_config:
            self.correct_dimensions = base_bin_config['dim']
        else:
            self.correct_dimensions = self.default_dimensions
        self.drop_bin = BaseBin(base_bin_config, self.pb_instance)

    def test_init(self):
        correct_mesh_scale = self.default_mesh_scale
        assert self.drop_bin.mesh_scale == correct_mesh_scale
        assert self.drop_bin.bin_origin == self.correct_bin_origin
        assert self.drop_bin.dimensions == correct_dimensions
        assert self.drop_bin.bin_orientation == self.correct_bin_orientation
        assert self.drop_bin.bin_id > -1
        assert self.drop_bin.constraint_id > -1

    def test_create_collision_and_visual_shape(self):
        frame_offset = [0, 0, 0]
        collision_shape, visual_shape = self.drop_bin.create_collision_and_visual_shape(
            file_name=self.drop_bin.bin_file_name,
            mesh_scale=self.drop_bin.mesh_scale,
            frame_offset=fame_offset,
            load_bin=True
        )
        assert collision_shape > -1
        assert visual_shape > -1
    
    def test_create_constraint(self):
        """
        Tests the create_constraint method of BaseBin. Ensures the constraint for the drop bin has been successfully
        created with the proper values.
        """
        assert self.drop_bin.constraint_id > -1
        base, no_body = -1, -1
        origin, aligned = [0, 0, 0], [0, 0, 0]
        upright_orientation = [0, 0, 0, 1]
        constraint_info = pb.getConstraintInfo(self.drop_bin.constraint_id)
        correct_parent_body_unique_id = self.drop_bin.bin_id
        assert constraint_info[0] == correct_parent_body_unique_id
        correct_parent_link_index = base
        assert constraint_info[1] == correct_parent_link_index
        correct_child_body_unique_id = no_body
        assert constraint_info[2] == correct_child_body_unique_id
        correct_child_link_index = base
        assert constraint_info[3] == correct_child_link_index
        correct_constraint_type = pb.JOINT_FIXED
        assert constraint_info[4] == correct_constraint_type
        correct_joint_axis = aligned
        assert constraint_info[5] == correct_joint_axis
        correct_parent_frame_position = origin
        assert constraint_info[6] == correct_parent_frame_position
        correct_child_frame_position = self.correct_bin_origin
        assert constraint_info[7] == correct_child_frame_position
        correct_parent_frame_orientation = upright_orientation
        assert constraint_info[8] == correct_parent_frame_orientation
        correct_child_frame_orientation = self.correct_bin_orientation
        assert constraint_info[9] == correct_child_frame_orientation

    def test_load_bin(self):
        """
        Tests the load_bin method of BinBase. Ensures the bin picking bin is successfully loaded into the simulation.
        """
        assert self.drop_bin.bin_id > -1
        drop_pin_position, drop_bin_orientation = pb.getBasePositionAndOrientation(self.drop_bin.bin_id)
        assert drop_pin_position == self.correct_bin_origin
        assert drop_bin_orientation == self.correct_bin_orientation
    
    def test_reset(self):
        """
        Tests the reset method of BinBase. Ensures that bin is successfully reloded into the simulation.
        """
        pb.resetSimulation()
        self.drop_bin.reset()
        assert self.drop_bin.bin_id > -1
        drop_pin_position, drop_bin_orientation = pb.getBasePositionAndOrientation(self.drop_bin.bin_id)
        assert drop_pin_position == self.correct_bin_origin
        assert drop_bin_orientation == self.correct_bin_orientation
