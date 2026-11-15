"""The prepared owner model actually moves both physical grippers with a fixed base."""
from pathlib import Path
import json

import mujoco
import numpy as np

from sx_embodiments.kinematics import Kinematics
from sx_embodiments.parts import GripperSpec
from tools.prepare_stationary_benchmark import prepare_stationary_benchmark


def test_fixed_pair_preserves_owner_fk_and_aperture_during_cpu_physics(tmp_path, monkeypatch):
    output = tmp_path/'prepared'
    body, spec = prepare_stationary_benchmark(output)
    monkeypatch.setenv('SX_EMBODIMENTS_ASSETS',str(output/'assets'))
    model = mujoco.MjModel.from_xml_path(str(output/'robot/scene.xml'))
    data = mujoco.MjData(model)
    names = tuple(body.joint_names)
    qadr = np.asarray([model.jnt_qposadr[model.joint(name).id] for name in names])
    controls = np.asarray([np.flatnonzero(model.actuator_trnid[:,0] == model.joint(name).id).item() for name in names])
    assert not np.any(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    assert model.nu == 14 and model.njnt == 16 and model.neq == 2
    q = np.asarray(spec['nominal_q']).copy()
    q[0] += .07
    q[7] -= .06
    data.qpos[qadr] = q
    values = dict(zip(names,q,strict=True))
    for dependent in spec['dependent_joints']:
        data.qpos[model.jnt_qposadr[model.joint(dependent['joint']).id]] = dependent['multiplier']*values[dependent['leader']]+dependent['offset']
    data.ctrl[controls] = q
    mujoco.mj_forward(model,data)
    kin = Kinematics.of(body.urdf_bytes)
    root = next(iter(kin.links-kin.parent_joint.keys()))
    for link in kin.links:
        pose = np.asarray(kin.pose(root,link,values)).reshape(4,4)
        np.testing.assert_allclose(data.xpos[model.body(link).id],pose[:3,3],atol=2e-12)
        np.testing.assert_allclose(data.xmat[model.body(link).id].reshape(3,3),pose[:3,:3],atol=2e-12)
    for site in body.sites:
        pose = np.asarray(site.transform(kin,root,values)).reshape(4,4)
        np.testing.assert_allclose(data.site_xpos[model.site(site.name).id],pose[:3,3],atol=2e-12)
        np.testing.assert_allclose(data.site_xmat[model.site(site.name).id].reshape(3,3),pose[:3,:3],atol=2e-12)
    root_position = data.xpos[model.body(root).id].copy()
    components = {str(c.component_id):c for c in body.components}
    targets = {}
    for role, aperture in spec['apertures'].items():
        part = components[aperture['component_id']].part
        assert isinstance(part,GripperSpec)
        width = .05 if role == 'left_hand' else .02
        index = names.index(aperture['drive_joint'])
        q[index] = part.drive_from_aperture(width)
        targets[role] = (part,index,width)
    q[0] += .04
    q[7] -= .03
    data.ctrl[controls] = q
    initial_hand = data.site_xpos.copy()
    for _ in range(500):
        mujoco.mj_step(model,data)
    mujoco.mj_forward(model,data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert not any(w.number for w in data.warning)
    np.testing.assert_array_equal(data.xpos[model.body(root).id],root_position)
    assert np.max(np.linalg.norm(data.site_xpos-initial_hand,axis=1)) > .005
    for part,index,width in targets.values():
        assert abs(part.aperture_from_drive(float(data.qpos[qadr[index]]))-width) < .001
    for dependent in spec['dependent_joints']:
        follower = data.qpos[model.jnt_qposadr[model.joint(dependent['joint']).id]]
        leader = data.qpos[qadr[names.index(dependent['leader'])]]
        assert abs(follower-dependent['multiplier']*leader-dependent['offset']) < 1e-4
    assert json.loads((output/'robot/receipt.json').read_text())['profile']['root_mode'] == 'fixed'
    assert Path(output/'assets').is_dir()
