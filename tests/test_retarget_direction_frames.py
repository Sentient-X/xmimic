"""Observed point-pair directions survive site frames and single-frame projection."""

from dataclasses import replace
import math

import numpy as np
import pytest
from sx_actions.urr import Motion, MotionFrame, Position
from sx_contracts import ComponentId
from sx_embodiments.assembly import Pose as SitePose
from sx_embodiments.sites import BodySite
from sx_geometry import Pose

from test_retarget_target import fixture_body
from xmimic.retarget_motion import motion_directions
from xmimic.retarget_target import (
    Direction, Disposition, IKConfig, TargetKind, TargetProfile, prepare_target, solve_sequence,
)


def _rotation(axis, angle):
    """Independent elementary rotation oracle."""
    c, s = math.cos(angle), math.sin(angle)
    if axis == 0:
        return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    if axis == 1:
        return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _body_with_sites(tmp_path, monkeypatch, separation):
    body = fixture_body(tmp_path, monkeypatch)
    origin = np.array([.2, -.1, .07])
    # Both are actual canonical points. Their unrelated orientation frames do
    # not rotate the displacement between their positions.
    return replace(body, sites=(
        BodySite('probe_k7', ComponentId('arm'), 'tip',
                 SitePose(tuple(origin), (.7, -.2, 1.1))),
        BodySite('marker_z9', ComponentId('arm'), 'tip',
                 SitePose(tuple(origin+separation), (-.4, .8, -.6))),
    ))


def _unit(target, q, root_p, root_quat):
    parent, child = target.direction_indices[0]
    poses = target.poses(q, root_p, root_quat)
    vector = poses[child][0] - poses[parent][0]
    return vector / np.linalg.norm(vector)


@pytest.mark.parametrize('local_vector', [(1., 0., 0.), (0., -1., 0.),
                                  tuple(float(x) for x in np.array([1., 2., -1.])/math.sqrt(6))])
def test_offset_site_pair_and_jacobian_follow_independent_geometry(tmp_path, monkeypatch, local_vector):
    local_direction = _rotation(2, .4) @ _rotation(1, -.3) @ _rotation(0, .2) @ np.asarray(local_vector)
    body = _body_with_sites(tmp_path, monkeypatch, .37*local_direction)
    mapping = Direction('observation_17', 'observation_91', 'probe_k7', 'marker_z9', kind=TargetKind.SITE)
    target = prepare_target(body, TargetProfile((), holds=(('q3', .3),), directions=(mapping,)),
                            expected_embodiment_id=str(body.id))
    q, root_p = np.array([.6, .3, .3]), np.array([.4, -.8, 1.2])
    root_quat = np.array([math.cos(.25), 0., 0., math.sin(.25)])
    expected = _rotation(2, .5+q[0]+q[1]) @ local_direction
    unit = _unit(target, q, root_p, root_quat)
    np.testing.assert_allclose(unit, expected, atol=1e-14)
    np.testing.assert_allclose(_unit(target, q, root_p+[8., -3., 2.], root_quat), expected, atol=1e-14)
    parent, child = target.direction_indices[0]
    poses, spatial = target.poses(q, root_p, root_quat), target.jacobians(q, root_quat)
    length = np.linalg.norm(poses[child][0]-poses[parent][0])
    analytic = (np.eye(3)-np.outer(unit, unit))/length @ (spatial[child][:3]-spatial[parent][:3])
    epsilon = 1e-6
    finite = np.column_stack([
        (_unit(target, q+delta, root_p, root_quat)-_unit(target, q-delta, root_p, root_quat))/(2*epsilon)
        for delta in np.eye(3)*epsilon
    ])
    np.testing.assert_allclose(analytic, finite, atol=1e-9)
    np.testing.assert_allclose(analytic[:, 2], 0., atol=1e-14)


@pytest.mark.parametrize('source_local_sign', [1., -1.])
def test_observed_heading_cannot_be_redefined_by_mirrored_source_or_robot_reference(
        tmp_path, monkeypatch, source_local_sign):
    body = _body_with_sites(tmp_path, monkeypatch, .37*_rotation(2, .3) @ np.array([1., 0., 0.]))
    mapping = Direction('trace_p', 'trace_q', 'probe_k7', 'marker_z9', kind=TargetKind.SITE)
    # The same observed world vector may arise from opposite local skeleton axes.
    source_yaw = 0. if source_local_sign > 0 else math.pi
    origin = np.array([3., 4., 5.])
    observed = _rotation(2, source_yaw) @ np.array([source_local_sign, 0., 0.])
    motion = Motion((MotionFrame(0, 'capture/world', {
        'trace_p': Pose(origin, np.array([math.cos(source_yaw/2), 0., 0., math.sin(source_yaw/2)])),
        'trace_q': Position(origin+.14*observed),
    }),))
    source_id = motion.id
    desired = motion_directions(motion, {mapping.key: ('trace_p', 'trace_q')})
    retained = desired[mapping.key].copy()
    np.testing.assert_allclose(retained, [[1., 0., 0.]], atol=1e-14)
    profile = TargetProfile((), holds=(('q2', .1), ('q3', .3)), directions=(mapping,))
    args = dict(positions={}, directions=desired, root_p=np.array([[.2, -.3, .4]]),
                root_quat=np.array([[math.cos(.2), 0., 0., math.sin(.2)]]),
                fps=30., initial_q=[.1, .5, .3], config=IKConfig(direction_tolerance_rad=1e-6))
    target = prepare_target(body, profile, expected_embodiment_id=str(body.id))
    result = solve_sequence(target, **args)
    assert result.feasible and len(result.q) == 1
    # .4 root yaw + .3 point-pair yaw + .1 held coordinate require q1=-.8.
    np.testing.assert_allclose(result.q[0], [.1, -.8, .3], atol=2e-6)
    np.testing.assert_allclose(_unit(target, result.q[0], args['root_p'][0], args['root_quat'][0]),
                               retained[0], atol=1e-6)
    # A wrong fixed reference must fail on the very first sample. Fitting an
    # orientation offset to that reference would manufacture zero error here.
    fixed = prepare_target(body, replace(profile, holds=profile.holds+(('q1', .5),)),
                           expected_embodiment_id=str(body.id))
    rejected = solve_sequence(fixed, **args)
    assert not rejected.feasible
    assert rejected.dispositions == (Disposition.RESIDUAL_EXCEEDED,)
    np.testing.assert_allclose(rejected.direction_residual_rad, [[1.3]], atol=1e-14)
    np.testing.assert_array_equal(desired[mapping.key], retained)
    assert not desired[mapping.key].flags.writeable
    assert motion.id == source_id
