"""Independent MuJoCo geometry checks for mechanically derived nominal bundles."""
import hashlib
import json
from dataclasses import replace

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from sx_contracts.identity import content_digest

from xmimic.embodiment_bundle import TrainingProfile, adapt_nominal_bundle
from xmimic.kin import Kinematics
from xmimic.model import load_model


def fixture(tmp_path):
    scene = '''<mujoco><compiler angle="radian" eulerseq="XYZ"/><option timestep=".005"/>
    <worldbody><geom name="floor" type="plane" size="2 2 .1"/>
    <body name="base" pos="0 0 .6"><freejoint name="floating_root"/><geom size=".1"/>
      <body name="upper" pos=".2 -.1 0" euler=".2 -.3 .4">
        <joint name="bend" axis="0 1 0" range="-1 1"/><geom size=".05"/>
        <body name="fixed" pos=".1 .02 -.2" euler=".1 .2 .3"><geom size=".03"/>
          <body name="sole" pos="0 0 -.1"><joint name="twist" axis="0 0 1" range="-.7 .8"/>
          <geom size=".04"/></body></body></body></body></worldbody>
    <actuator><position joint="bend" kp="10" kv="1"/>
      <position joint="twist" kp="9" kv=".8"/></actuator></mujoco>'''
    (tmp_path / 'scene.xml').write_text(scene)
    nominal = {'expected_embodiment_id': 'fixture-only', 'joints': [{'name': n, 'default_position': .1, 'action_scale': .3}
                          for n in ('twist', 'bend')],
               'tracked_links': ['base', 'fixed', 'sole'], 'contacts': [{'link': 'sole', 'height_m': .04}]}
    receipt = {'schema': 'sx-embodiments.nominal-mujoco/v2',
               'rotation_convention': 'urdf-rpy-extrinsic-XYZ', 'embodiment_id': 'fixture-only', 'scene_sha256': hashlib.sha256(scene.encode()).hexdigest(),
               'joint_names': ['twist', 'bend'], 'profile': nominal,
               'profile_sha256': str(content_digest(nominal)),
               'files': [['scene.xml', hashlib.sha256(scene.encode()).hexdigest()]]}
    (tmp_path / 'receipt.json').write_text(json.dumps(receipt))
    profile = TrainingProfile('base', ((0., 0.),) * 3, (.8, 1.1), 0., (0.,) * 3,
                              (0.,) * 3, (0.,) * 3, (0.,) * 3, 0., 0., 32, 16)
    return profile


def test_derived_native_order_and_full_tree_match_mujoco(tmp_path):
    meta = adapt_nominal_bundle(tmp_path, fixture(tmp_path))
    model, _ = load_model(tmp_path)
    assert meta['qpos_adr'] == [8, 7]
    assert meta['actuator_of_joint'] == [1, 0]
    assert meta['kin_joint_idx'] == [-1, 1, -1, 0]
    assert 'mirror_permutation' not in meta
    kin = Kinematics(meta)
    data = mujoco.MjData(model)
    root = np.array([.3, -.4, .7])
    quat = np.array([.5, .5, -.5, .5])
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    for q in ([.2, -.4], [-.6, .8]):
        data.qpos[:3], data.qpos[3:7], data.qpos[meta['qpos_adr']] = root, quat, q
        mujoco.mj_forward(model, data)
        p, r, _, _ = kin.forward(jnp.asarray(root), jnp.asarray(rotation.reshape(3, 3)),
                                 jnp.zeros(3), jnp.zeros(3), jnp.asarray(q), jnp.zeros(2))
        np.testing.assert_allclose(p, data.xpos[meta['tracked_body_ids']], atol=2e-7)
        np.testing.assert_allclose(r, data.xmat[meta['tracked_body_ids']].reshape(-1, 3, 3), atol=2e-7)
    with pytest.raises(FileExistsError):
        adapt_nominal_bundle(tmp_path, fixture(tmp_path))


@pytest.mark.parametrize('change', [{'friction_range': (1., 1.)}, {'com_body': 'missing'},
                                    {'reset_pos': (-1., 0., 0.)}, {'nconmax': 0},
                                    {'mirror_permutation': (1, 0)}, {'com_range': ((1., 0.),) * 3}])
def test_invalid_training_assumptions_refused_before_metadata(tmp_path, change):
    profile = fixture(tmp_path)
    with pytest.raises(ValueError):
        adapt_nominal_bundle(tmp_path, replace(profile, **change))
    assert not (tmp_path / 'meta.json').exists()


def test_scene_receipt_tampering_refused(tmp_path):
    profile = fixture(tmp_path)
    with (tmp_path / 'scene.xml').open('a') as f:
        f.write(' ')
    with pytest.raises(ValueError, match='receipt'):
        adapt_nominal_bundle(tmp_path, profile)


def test_legacy_or_wrong_rotation_export_cannot_enter_training(tmp_path):
    from xmimic.embodiment_bundle import verify_nominal_receipt
    fixture(tmp_path)
    path = tmp_path/'receipt.json'
    receipt = json.loads(path.read_text())
    receipt['schema'] = 'sx-embodiments.nominal-mujoco/v1'
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='regenerate'):
        verify_nominal_receipt(tmp_path)
    receipt['schema'] = 'sx-embodiments.nominal-mujoco/v2'
    scene = (tmp_path/'scene.xml').read_text().replace('eulerseq="XYZ"', 'eulerseq="xyz"')
    (tmp_path/'scene.xml').write_text(scene)
    digest = hashlib.sha256(scene.encode()).hexdigest()
    receipt['scene_sha256'] = digest
    receipt['files'] = [['scene.xml', digest]]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='extrinsic RPY'):
        verify_nominal_receipt(tmp_path)
