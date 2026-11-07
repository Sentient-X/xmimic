"""Independent planar geometry oracle for canonical target projection and CPU IK."""
from dataclasses import replace
import json
import math

import numpy as np
import pytest
from sx_contracts import ComponentId
from sx_contracts.assets import AssetFormat, AssetProvenance, AssetRef, AssetRole, ProvenancedAsset
from sx_embodiments import Embodiment
from sx_embodiments.assembly import Pose
from sx_embodiments.compose import RootMount, body_component
from sx_embodiments.identity import EmbodimentKind, EmbodimentName, Lineage, PartId
from sx_embodiments.layout import Bounds, CoordinateUnit, JointAxis, JointLayout
from sx_embodiments.parts import ChainSpec, Limb
from sx_embodiments.sites import BodySite

from xmimic.retarget_target import (
    Disposition, IKConfig, Landmark, RetargetError, TargetKind, TargetProfile,
    prepare_target, solve_sequence, training_preflight,
)


def fixture_body(tmp_path, monkeypatch, passive=False):
    links = ''.join(f'<link name="{name}"/>' for name in ('base', 'a', 'b', 'tip', 'held', 'rod'))
    def joint(name, parent, child, xyz='0 0 0'):
        return f'<joint name="{name}" type="revolute"><parent link="{parent}"/><child link="{child}"/><origin xyz="{xyz}"/><axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="10" velocity="2"/></joint>'
    xml = ('<robot name="analytic">'+links+joint('q1', 'base', 'a')+joint('q2', 'a', 'b', '1 0 0')
           +joint('q3', 'base', 'held')+'<joint name="tip_fixed" type="fixed"><parent link="b"/><child link="tip"/><origin xyz="1 0 0"/></joint>'
           +(joint('rod_joint', 'held', 'rod') if passive else '<joint name="rod_fixed" type="fixed"><parent link="held"/><child link="rod"/></joint>')+'</robot>').encode()
    (tmp_path / 'robot.urdf').write_bytes(xml)
    monkeypatch.setenv('SX_EMBODIMENTS_ASSETS', str(tmp_path))
    asset = ProvenancedAsset(AssetRef.from_bytes(xml, uri='package://sx-embodiments/robot.urdf', format=AssetFormat.URDF, role=AssetRole.DESCRIPTION),
                            AssetProvenance('https://example.com/independent-fixture', 'v1', 'robot.urdf', 'MIT'))
    # Deliberately different from URDF document order.
    names = ('q2', 'q1', 'q3')
    part = ChainSpec(PartId('analytic'), JointLayout(tuple(JointAxis(n, CoordinateUnit.RADIAN, Bounds(-2, 2)) for n in names)), (0., 0., 0.))
    return Embodiment(EmbodimentName('analytic'), 'Analytic', EmbodimentKind.ROBOT, Lineage('fixture'),
                      (body_component('arm', part, RootMount('base'), limb=Limb.ARM),), (asset,),
                      sites=(BodySite('tip_site', ComponentId('arm'), 'tip', Pose((.2, 0., 0.), (0., 0., 0.))),))


def profile(**kwargs):
    return TargetProfile((Landmark('hand', 'tip_site', orientation=True),), holds=(('q3', .3),), **kwargs)


def analytic(q1, q2, yaw, translation):
    angle = q1 + q2 + yaw
    p = np.asarray(translation) + [math.cos(q1+yaw)+1.2*math.cos(angle), math.sin(q1+yaw)+1.2*math.sin(angle), 0.]
    c, s = math.cos(angle), math.sin(angle)
    return p, np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def inputs():
    q1 = np.linspace(.2, .5, 8); q2 = np.linspace(.4, .6, 8)
    yaw = np.linspace(-.3, .2, 8)
    root_p = np.column_stack((np.linspace(.1, .3, 8), np.zeros(8), np.linspace(.5, .8, 8)))
    root_q = np.column_stack((np.cos(yaw/2), np.zeros(8), np.zeros(8), np.sin(yaw/2)))
    poses = [analytic(a,b,y,p) for a,b,y,p in zip(q1,q2,yaw,root_p)]
    return dict(positions={'hand': np.array([v[0] for v in poses])}, orientations={'hand': np.array([v[1] for v in poses])},
                root_p=root_p, root_quat=root_q, fps=30., initial_q=[.1, .1, .3]), q1, q2


def test_complete_sequence_matches_independent_geometry_and_native_order(tmp_path, monkeypatch):
    body = fixture_body(tmp_path, monkeypatch)
    target = prepare_target(body, profile(), expected_embodiment_id=str(body.id))
    args, q1, q2 = inputs()
    result = solve_sequence(target, **args, config=IKConfig(position_tolerance_m=1e-6, orientation_tolerance_rad=1e-6))
    assert result.feasible
    np.testing.assert_allclose(result.q[:, 0], q2, atol=1e-5)
    np.testing.assert_allclose(result.q[:, 1], q1, atol=1e-5)
    np.testing.assert_array_equal(result.q[:, 2], .3)
    assert target.binding.joint_names == ('q2', 'q1', 'q3')
    assert not result.q.flags.writeable
    path = tmp_path / 'motion.npz'
    np.savez(path, **result.npz_arrays())
    with np.load(path, allow_pickle=False) as motion:
        assert motion['joint_names'].tolist() == list(body.joint_names)
        assert motion['embodiment_id'].item() == str(body.id)
    json.dumps(result.provenance(), allow_nan=False)


def test_unreachable_frame_is_recorded_without_cutting_later_frames(tmp_path, monkeypatch):
    body = fixture_body(tmp_path, monkeypatch)
    target = prepare_target(body, profile(), expected_embodiment_id=str(body.id))
    args, _, _ = inputs()
    args['positions']['hand'][2] += [100, 0, 0]
    result = solve_sequence(target, **args)
    assert len(result.q) == 8
    assert result.dispositions[2] is Disposition.RESIDUAL_EXCEEDED
    assert result.dispositions[-1] is Disposition.GEOMETRICALLY_FEASIBLE
    assert np.all(result.q >= -2) and np.all(result.q <= 2)
    with pytest.raises(RetargetError, match='Infeasible'):
        result.npz_arrays()


def test_passive_and_unmatched_coordinates_require_explicit_accounting(tmp_path, monkeypatch):
    body = fixture_body(tmp_path, monkeypatch, passive=True)
    with pytest.raises(RetargetError, match='passive'):
        prepare_target(body, profile(), expected_embodiment_id=str(body.id))
    p = profile(nominal_passive=(('rod_joint', .1),))
    target = prepare_target(body, p, expected_embodiment_id=str(body.id))
    assert target.binding.profile.nominal_passive == (('rod_joint', .1),)
    with pytest.raises(RetargetError, match='Unmatched'):
        prepare_target(body, replace(p, holds=()), expected_embodiment_id=str(body.id))
    with pytest.raises(RetargetError, match='outside'):
        prepare_target(body, replace(p, nominal_passive=(('rod_joint', 3),)), expected_embodiment_id=str(body.id))
    other = prepare_target(body, replace(p, holds=(('q3', .4),)), expected_embodiment_id=str(body.id))
    assert target.binding.sha256 != other.binding.sha256
    with pytest.raises(RetargetError, match='identity'):
        prepare_target(body, p, expected_embodiment_id='0'*64)


@pytest.mark.parametrize('fault', ['nan', 'quat', 'reflection', 'hold', 'mapping'])
def test_invalid_inputs_refuse_before_solving(tmp_path, monkeypatch, fault):
    body = fixture_body(tmp_path, monkeypatch)
    target = prepare_target(body, profile(), expected_embodiment_id=str(body.id))
    args, _, _ = inputs()
    if fault == 'nan': args['positions']['hand'][0, 0] = np.nan
    if fault == 'quat': args['root_quat'][0] *= 2
    if fault == 'reflection': args['orientations']['hand'][0] *= -1
    if fault == 'hold': args['initial_q'][2] = 0
    if fault == 'mapping': args['positions']['other'] = args['positions'].pop('hand')
    with pytest.raises(RetargetError): solve_sequence(target, **args)


def test_preflight_reports_owner_gaps_without_qualification(tmp_path, monkeypatch):
    body = fixture_body(tmp_path, monkeypatch, passive=True)
    result = training_preflight(body)
    assert {m.coordinate for m in result.missing_facts if m.code == 'undocumented_drive'} == {'q1', 'q2', 'q3'}
    assert any(m.coordinate == 'rod_joint' for m in result.missing_facts)
    assert not result.to_dict()['training_ready']
    assert any(m.code == 'missing_mjcf' for m in result.missing_facts)


def test_analytic_site_jacobian_matches_independent_finite_difference(tmp_path, monkeypatch):
    body = fixture_body(tmp_path, monkeypatch)
    target = prepare_target(body, profile(), expected_embodiment_id=str(body.id))
    q = np.array([.6, .3, .3]); yaw = .4
    root_q = np.array([math.cos(yaw/2),0,0,math.sin(yaw/2)])
    jac = target.jacobians(q, root_q)[0]
    # Independent planar equations, including the site's nonzero lever arm.
    theta = q[0]+q[1]+yaw
    expected = np.array([[-1.2*math.sin(theta),-math.sin(q[1]+yaw)-1.2*math.sin(theta),0],
                         [1.2*math.cos(theta),math.cos(q[1]+yaw)+1.2*math.cos(theta),0],
                         [0,0,0],[0,0,0],[0,0,0],[1,1,0]])
    np.testing.assert_allclose(jac, expected, atol=1e-12)


def test_coupled_nonplanar_orientation_solver(tmp_path, monkeypatch):
    body = fixture_body(tmp_path, monkeypatch)
    # Independent X elbow axis makes this a spatial orientation oracle.
    original = body.urdf_bytes
    xml = original.replace(b'<joint name="q2" type="revolute"><parent link="a"/><child link="b"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/>',
                           b'<joint name="q2" type="revolute"><parent link="a"/><child link="b"/><origin xyz="1 0 0"/><axis xyz="1 0 0"/>')
    assert xml != original
    (tmp_path/'robot.urdf').write_bytes(xml)
    asset = replace(body.assets[0], asset=AssetRef.from_bytes(xml, uri='package://sx-embodiments/robot.urdf', format=AssetFormat.URDF, role=AssetRole.DESCRIPTION))
    body = replace(body, assets=(asset,))
    target = prepare_target(body, profile(), expected_embodiment_id=str(body.id))
    a, b, yaw = .8, 1.2, -.3
    c, s = math.cos(yaw), math.sin(yaw)
    root_R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    ca,sa,cb,sb = math.cos(a),math.sin(a),math.cos(b),math.sin(b)
    expected_R = root_R @ np.array([[ca,-sa,0],[sa,ca,0],[0,0,1.]]) @ np.array([[1.,0,0],[0,cb,-sb],[0,sb,cb]])
    expected_p = root_R @ np.array([2.2*ca,2.2*sa,0.])
    result = solve_sequence(target, {'hand': np.tile(expected_p,(2,1))},
                            orientations={'hand':np.tile(expected_R,(2,1,1))}, root_p=np.zeros((2,3)),
                            root_quat=np.tile([math.cos(yaw/2),0,math.sin(yaw/2),0],(2,1)),
                            fps=30., initial_q=[0.,0.,.3],config=IKConfig(position_tolerance_m=1e-6,orientation_tolerance_rad=1e-6))
    assert result.feasible
    np.testing.assert_allclose(result.q, [[b,a,.3],[b,a,.3]], atol=1e-5)
