"""Source geometry survives the actual shared URR contract before body binding."""

import numpy as np
import pytest
from sx_actions.urr import Motion, MotionError, MotionFrame, Position, motion_from_dict, motion_to_dict
from sx_geometry import Pose

from test_retarget_source import ROWS, fixture
from xmimic.retarget_motion import (
    motion_directions, motion_from_clip, motion_orientations, motion_positions, soma_motion,
)
from xmimic.retarget_source import SourceError, parse_soma_bvh


def test_observed_poses_native_time_and_shared_roundtrip_are_preserved():
    world = np.array([[1, 0, 0, 4], [0, 0, -1, 5], [0, 1, 0, 6], [0, 0, 0, 1]], dtype=float)
    clip = parse_soma_bvh(fixture(), world_from_source=world)
    roles = {'pelvis': 'Hips', 'left_hand': 'Hand'}
    motion = motion_from_clip(clip, roles=roles, frame='capture/world', start_time_ns=123)
    assert isinstance(motion, Motion)
    assert [sample.time_ns for sample in motion.frames] == [123, 8_333_123]
    assert all(sample.frame == 'capture/world' and not sample.apertures for sample in motion.frames)
    for index, sample in enumerate(motion.frames):
        for role, joint in roles.items():
            pose = sample.poses[role]
            assert isinstance(pose, Pose)
            np.testing.assert_array_equal(pose.position, clip.world_positions[index, clip.joint_index(joint)])
            np.testing.assert_allclose(pose.to_matrix()[:3, :3], clip.world_rotations[index, clip.joint_index(joint)], atol=1e-14)
    # Independent metric placement oracle: no root subtraction, limb scaling or retiming.
    np.testing.assert_array_equal(motion.frames[1].poses['left_hand'].position, [6, 5, 8])
    assert motion_from_dict(motion_to_dict(motion)).id == motion.id
    assert soma_motion(fixture(), roles=roles, frame='capture/world', start_time_ns=123,
                       world_from_source=world).id == motion.id
    roles['left_hand'] = 'Root'
    with pytest.raises(TypeError):
        motion.frames[0].poses['invented'] = Position(np.zeros(3))
    with pytest.raises(ValueError):
        motion.frames[0].poses['left_hand'].position[0] = 0


def test_fractional_nanosecond_period_is_rounded_per_sample_without_drift():
    motion = soma_motion(fixture(rows=ROWS*2, count=4, dt='0.0333333333333333'),
                         roles={'hand': 'Hand'}, frame='world', start_time_ns=0,
                         world_from_source=np.eye(4))
    assert [sample.time_ns for sample in motion.frames] == [0, 33_333_333, 66_666_667, 100_000_000]
    with pytest.raises(MotionError, match='nanosecond resolution'):
        soma_motion(fixture(dt='1e-10'), roles={'hand': 'Hand'}, frame='world', start_time_ns=0,
                    world_from_source=np.eye(4))


def test_explicit_roles_lower_positions_orientations_and_directions_without_mutation():
    motion = soma_motion(fixture(), roles={'elbow': 'Hips', 'wrist': 'Hand'},
                         frame='source/world', start_time_ns=0, world_from_source=np.eye(4))
    before = motion.id
    points = motion_positions(motion, ('wrist',))
    rotations = motion_orientations(motion, ('wrist',))
    directions = motion_directions(motion, {'forearm': ('elbow', 'wrist')})
    np.testing.assert_allclose(directions['forearm'], [[0, 0, -1], [1, 0, 0]], atol=1e-14)
    np.testing.assert_array_equal(points['wrist'][1], [2, 2, 0])
    np.testing.assert_array_equal(rotations['wrist'][1], np.eye(3))
    assert motion.id == before
    assert all(not array.flags.writeable for values in (points, rotations, directions) for array in values.values())


def test_sparse_position_evidence_never_becomes_orientation_or_missing_targets():
    point = Position(np.array([1., 2., 3.]))
    motion = Motion((MotionFrame(0, 'world', {'point': point, 'same': point}),
                     MotionFrame(10, 'world', {'point': point, 'same': point})))
    np.testing.assert_array_equal(motion_positions(motion, ('point',))['point'], [[1, 2, 3], [1, 2, 3]])
    with pytest.raises(MotionError, match='orientation'):
        motion_orientations(motion, ('point',))
    with pytest.raises(MotionError, match='position'):
        motion_positions(motion, ('absent',))
    with pytest.raises(MotionError, match='coincident'):
        motion_directions(motion, {'bone': ('point', 'same')})
    sparse = Motion((motion.frames[0], MotionFrame(10, 'world', {})))
    with pytest.raises(MotionError, match='position'):
        motion_positions(sparse, ('point',))


@pytest.mark.parametrize('change', [dict(roles={}), dict(frame=''), dict(start_time_ns=-1), dict(start_time_ns=True)])
def test_implicit_correspondence_frame_or_epoch_is_refused(change):
    args = dict(roles={'hand': 'Hand'}, frame='world', start_time_ns=0, world_from_source=np.eye(4))
    with pytest.raises(MotionError):
        soma_motion(fixture(), **(args | change))
    with pytest.raises(SourceError):
        soma_motion(fixture(), **(args | {'roles': {'hand': 'missing'}}))
