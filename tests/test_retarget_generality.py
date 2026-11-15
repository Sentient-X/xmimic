"""One geometric engine, arbitrary names/order, and single or branched mechanisms.

These are numeric acceptance tests, not proof that shared solver ownership or
physical admission has been consolidated.
"""

import math

import numpy as np
import pytest
from sx_actions.urr import Motion, MotionBinding, MotionFrame, Position
from sx_contracts import ComponentId
from sx_contracts.assets import AssetFormat, AssetProvenance, AssetRef, AssetRole, ProvenancedAsset
from sx_embodiments import Embodiment
from sx_embodiments.assembly import Pose as SitePose
from sx_embodiments.compose import RootMount, body_component
from sx_embodiments.identity import EmbodimentKind, EmbodimentName, Lineage, PartId
from sx_embodiments.layout import Bounds, CoordinateUnit, JointAxis, JointLayout
from sx_embodiments.parts import ChainSpec, Limb
from sx_embodiments.sites import BodySite
from sx_geometry import FramedTransform, Pose, quat_to_mat

from xmimic.retarget_motion import motion_directions, motion_orientations, motion_positions
from xmimic.retarget_target import (
    Direction, IKConfig, Landmark, RetargetError, TargetKind, TargetProfile, prepare_target,
    solve_sequence,
)


def _mechanism(tmp_path, monkeypatch, count, *, renamed=False):
    """Several two-hinge chains plus an unobserved branch, with no anatomical labels."""
    tmp_path.mkdir()
    names = {}
    def name(token):
        if token not in names:
            names[token] = f'z{900-len(names)*7}' if renamed else token
        return names[token]
    base = name('frame_origin')
    links = [f'<link name="{base}"/>']
    joints, components, sites, correspondence = [], [], [], {}
    expected, initial = {}, {}
    for index in range(count + 1):
        component = name(f'unit_{index}')
        axes = []
        for k in range(1 if index == count else 2):
            joint = name(f'coordinate_{index}_{k}')
            child = name(f'node_{index}_{k}')
            parent = base if k == 0 else name(f'node_{index}_0')
            xyz = f'0 {index*3} 0' if k == 0 else '1 0 0'
            links.append(f'<link name="{child}"/>')
            joints.append(f'<joint name="{joint}" type="revolute"><parent link="{parent}"/>'
                          f'<child link="{child}"/><origin xyz="{xyz}"/><axis xyz="0 0 1"/>'
                          '<limit lower="-2" upper="2" effort="10" velocity="2"/></joint>')
            axes.append(JointAxis(joint, CoordinateUnit.RADIAN, Bounds(-2, 2)))
            expected[joint] = .37 if index == count else (.25+.05*index if k == 0 else .55-.04*index)
            initial[joint] = .37 if index == count else .1
        if renamed:
            axes.reverse()
        part = ChainSpec(PartId(name(f'part_{index}')), JointLayout(tuple(axes)), (0.,)*len(axes))
        components.append(body_component(component, part, RootMount(base), limb=Limb.ARM))
        if index == count:
            continue
        for k, offset in ((0, 0.), (1, 0.), (2, 1.)):
            role, site = name(f'observation_{index}_{k}'), name(f'site_{index}_{k}')
            link = name(f'node_{index}_{0 if k == 0 else 1}')
            sites.append(BodySite(site, ComponentId(component), link, SitePose((offset, 0., 0.), (0., 0., 0.))))
            correspondence[role] = site
    if renamed:
        components.reverse()
        links.reverse()
        joints.reverse()
        sites.reverse()
    xml = ('<robot name="fixture">'+''.join(links+joints)+'</robot>').encode()
    (tmp_path/'model.urdf').write_bytes(xml)
    monkeypatch.setenv('SX_EMBODIMENTS_ASSETS', str(tmp_path))
    asset = ProvenancedAsset(
        AssetRef.from_bytes(xml, uri='package://sx-embodiments/model.urdf',
                           format=AssetFormat.URDF, role=AssetRole.DESCRIPTION),
        AssetProvenance('https://example.com/analytic', 'v1', 'model.urdf', 'MIT'))
    body = Embodiment(EmbodimentName('renamed' if renamed else 'original'), 'Analytic mechanism',
                       EmbodimentKind.ROBOT, Lineage('analytic'), tuple(components), (asset,), sites=tuple(sites))
    binding = MotionBinding(body, correspondence)
    reference = FramedTransform('experiment_world', base,
                                Pose(np.array([.2, -.3, .4]), np.array([math.cos(.15), 0., 0., math.sin(.15)])))
    observations = {}
    landmarks, directions = [], []
    for index in range(count):
        a, b = expected[name(f'coordinate_{index}_0')], expected[name(f'coordinate_{index}_1')]
        origin = np.array([0., index*3, 0.])
        elbow = origin + [math.cos(a), math.sin(a), 0.]
        tip = elbow + [math.cos(a+b), math.sin(a+b), 0.]
        roles = [name(f'observation_{index}_{k}') for k in range(3)]
        observations[roles[0]] = Position(reference.pose.transform_point(origin))
        observations[roles[1]] = Position(reference.pose.transform_point(elbow))
        observations[roles[2]] = reference.pose.compose(Pose(tip, np.array([math.cos((a+b)/2), 0., 0., math.sin((a+b)/2)])))
        landmarks.append(Landmark(roles[2], binding.sites[roles[2]], TargetKind.SITE, orientation=True))
        directions.append(Direction(roles[0], roles[1], binding.sites[roles[0]], binding.sites[roles[1]], kind=TargetKind.SITE))
    motion = Motion((MotionFrame(123_456, reference.parent, observations),))
    profile = TargetProfile(tuple(landmarks), holds=((name(f'coordinate_{count}_0'), .37),), directions=tuple(directions))
    target = prepare_target(body, profile, expected_embodiment_id=str(body.id))
    return names, body, binding, reference, motion, target, expected, initial


def _solve(case):
    names, body, binding, reference, motion, target, expected, initial = case
    saved = []
    roles = tuple(mapping.source for mapping in target.binding.profile.landmarks)
    result = solve_sequence(
        target, motion_positions(motion, roles), orientations=motion_orientations(motion, roles),
        directions=motion_directions(motion, {d.key: (d.source_parent, d.source_child) for d in target.binding.profile.directions}),
        root_p=reference.pose.position[None], root_quat=reference.pose.quaternion[None], fps=50.,
        initial_q=[initial[name] for name in body.joint_names], on_frame=saved.append,
        config=IKConfig(position_tolerance_m=1e-7, orientation_tolerance_rad=1e-7, direction_tolerance_rad=1e-7),
    )
    # A real single sample: no repeated frame to satisfy a training-duration check.
    assert result.q.shape == (1, len(body.joint_names))
    assert len(saved) == 1 and saved[0].index == 0
    assert result.feasible
    found = dict(zip(body.joint_names, result.q[0], strict=True))
    for name, value in expected.items():
        assert found[name] == pytest.approx(value, abs=1e-6)
    observed = binding.observe(time_ns=motion.frames[0].time_ns, positions=found, reference=reference)
    for role, wanted in motion.frames[0].poses.items():
        np.testing.assert_allclose(observed.poses[role].position, wanted.position, atol=1e-7)
        if isinstance(wanted, Pose):
            np.testing.assert_allclose(quat_to_mat(observed.poses[role].quaternion), quat_to_mat(wanted.quaternion), atol=1e-7)
    return {token: found[value] for token, value in names.items() if value in found}


@pytest.mark.parametrize('count', [1, 3])
def test_one_engine_fits_fixed_base_and_multiple_branches_from_one_observed_frame(tmp_path, monkeypatch, count):
    case = _mechanism(tmp_path/'mechanism', monkeypatch, count)
    found = _solve(case)
    assert found[f'coordinate_{count}_0'] == .37
    from dataclasses import replace
    with pytest.raises(RetargetError, match='Unmatched'):
        prepare_target(case[1], replace(case[5].binding.profile, holds=()), expected_embodiment_id=str(case[1].id))


def test_every_name_and_native_order_can_change_without_changing_physical_solution(tmp_path, monkeypatch):
    original = _mechanism(tmp_path/'original', monkeypatch, 3)
    original_order = original[1].joint_names
    before = _solve(original)
    renamed = _mechanism(tmp_path/'renamed', monkeypatch, 3, renamed=True)
    reverse_names = {value: token for token, value in renamed[0].items()}
    assert [reverse_names[n] for n in renamed[1].joint_names] != list(original_order)
    assert not set(original_order) & set(renamed[1].joint_names)
    after = _solve(renamed)
    assert before.keys() == after.keys()
    for token in before:
        assert before[token] == pytest.approx(after[token], abs=1e-10)
