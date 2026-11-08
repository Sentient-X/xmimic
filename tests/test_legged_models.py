"""CPU checks against MuJoCo for a small four-contact robot, not a qualified embodiment."""

import json

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from xmimic.algorithm import Mirror
from xmimic.kin import Kinematics
from xmimic.model import contact_indices, load_meta, load_model
from xmimic.sim import XiLayout


def bundle(tmp_path):
    limbs = ''.join(
        f'<body name="leg{i}" pos="{x} {y} 0"><joint name="j{i}" axis="0 1 0" range="-1 1"/>'
        f'<geom type="capsule" size=".02" fromto="0 0 0 0 0 -.2"/>'
        f'<body name="foot{i}" pos="0 0 -.2"><geom size=".03"/></body></body>'
        for i, (x, y) in enumerate(((.2, .2), (.2, -.2), (-.2, .2), (-.2, -.2)))
    )
    actuators = ''.join(f'<position joint="j{i}" kp="10" kv="1"/>' for i in range(4))
    xml = (f'<mujoco><compiler angle="radian"/><option timestep=".005"/>'
           f'<worldbody><body name="root" pos="0 0 .5"><freejoint/>'
           f'<geom size=".1"/>{limbs}</body></worldbody><actuator>{actuators}</actuator></mujoco>')
    (tmp_path / 'scene.xml').write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    sim_keys = ('timestep', 'integrator', 'cone', 'solver', 'iterations', 'tolerance',
                'ls_iterations', 'ls_tolerance', 'ccd_iterations', 'impratio')
    meta = {
        'embodiment_id': 'test-fixture-only',
        'joint_names': [f'j{i}' for i in range(4)],
        'qpos_adr': [7, 8, 9, 10], 'dof_adr': [6, 7, 8, 9],
        'actuator_of_joint': [0, 1, 2, 3],
        'default_joint_pos': [0.] * 4, 'action_scale': [.5] * 4,
        'q_lo': [-1.] * 4, 'q_hi': [1.] * 4,
        'free_qpos_adr': 0, 'free_dof_adr': 0,
        'tracked_bodies': [f'foot{i}' for i in range(4)],
        'contact_bodies': [f'foot{i}' for i in range(4)],
        'contact_heights': [.03] * 4,
        'sim': {key: getattr(model.opt, key) for key in sim_keys}, 'exact': {},
    }
    (tmp_path / 'meta.json').write_text(json.dumps(meta))
    return model, meta


def test_four_contacts_and_arbitrary_joint_names(tmp_path):
    _, expected = bundle(tmp_path)
    model, meta = load_model(tmp_path, motion_format='retargeted_npz')
    assert model.nu == 4
    assert meta['joint_names'] == expected['joint_names']
    assert contact_indices(meta) == [0, 1, 2, 3]


@pytest.mark.parametrize('field,value,message', [
    ('actuator_of_joint', [1, 0, 2, 3], 'actuator does not drive'),
    ('qpos_adr', [8, 7, 9, 10], 'state addresses disagree'),
    ('contact_bodies', ['missing'], 'contact_bodies'),
    ('action_scale', [0, 1, 1, 1], 'action_scale must be positive'),
    ('free_qpos_adr', 1, 'single floating root'),
    ('free_qpos_adr', 0.0, 'must be integers'),
    ('qpos_adr', [7., 8., 9., 10.], 'integer indices'),
])
def test_inconsistent_robot_bundle_is_refused(tmp_path, field, value, message):
    _, meta = bundle(tmp_path)
    meta[field] = value
    (tmp_path / 'meta.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=message):
        load_model(tmp_path)


def test_g1_csv_cannot_be_replayed_on_another_body(tmp_path):
    bundle(tmp_path)
    with pytest.raises(ValueError, match='G1-specific'):
        load_model(tmp_path, motion_format='lafan1_csv')


def test_unrepresented_moving_joint_is_refused(tmp_path):
    _, meta = bundle(tmp_path)
    for field in ('joint_names', 'qpos_adr', 'dof_adr', 'actuator_of_joint',
                  'default_joint_pos', 'action_scale', 'q_lo', 'q_hi'):
        meta[field] = meta[field][:-1]
    (tmp_path / 'meta.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='every moving coordinate'):
        load_model(tmp_path)


def test_every_batched_policy_must_match_identity_and_architecture():
    from xmimic.evaluate import check_policy_config

    first = dict(embodiment_id='a', goal_frames=10, hist=10, enc_hidden='8,8',
                 dec_hidden='8,8', motion_format='retargeted_npz')
    check_policy_config(dict(first), first)
    for field, value in [('embodiment_id', 'b'), ('hist', 8), ('enc_hidden', '8,4')]:
        with pytest.raises(ValueError, match=field):
            check_policy_config({**first, field: value}, first)


def test_fast_reward_uses_each_contact_height(tmp_path):
    from xmimic.fast import FastXMimic
    from xmimic.mdp import MDP
    from xmimic.motion import load_motion

    _, meta = bundle(tmp_path)
    kin = Kinematics({**meta, 'kin_parent': [-1, 0, 0, 0, 0],
                      'kin_joint_idx': [-1, 0, 1, 2, 3],
                      'kin_axes': [[0, 1, 0]] * 5,
                      'kin_off_p': [[0, 0, 0]] + [[0, 0, -.2]] * 4,
                      'kin_off_R': [np.eye(3).tolist()] * 5, 'kin_tracked_idx': [1, 2, 3, 4]})
    np.savez(tmp_path / 'clip.npz', root_p=np.tile([0., 0., .3], (4, 1)),
             root_quat=np.tile([1., 0., 0., 0.], (4, 1)), q=np.zeros((4, 4)), fps=50.,
             joint_names=np.asarray(meta['joint_names']), embodiment_id=np.asarray(meta['embodiment_id']))
    motion = load_motion(str(tmp_path), kin, [0, 1, 2, 3], motion_format='retargeted_npz',
                         joint_names=meta['joint_names'], embodiment_id=meta['embodiment_id'])
    mdp = MDP(kin, [0.] * 4, [0, 1, 2, 3], 8, contact_heights=[.03, .09, .11, .22])
    fast = FastXMimic.__new__(FastXMimic)
    fast.mdp = mdp
    frames = jnp.asarray([0, 1])
    state = motion.ref_state(frames)
    action = jnp.zeros((2, 4))
    expected = mdp.reward(motion, state, frames, action, action, w_contact=.25)
    actual, _ = fast._reward_fk(motion, state, frames, action, action, kin.forward(*state), .25)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_paper_renderer_refuses_another_embodiment_before_rendering(tmp_path, monkeypatch):
    from tools import render_gif

    source = tmp_path / 'different-robot.npz'
    np.savez(source, embodiment_id=np.asarray('another-body'))
    monkeypatch.setattr('sys.argv', ['render_gif', str(source)])
    with pytest.raises(ValueError, match='cannot render a different embodiment'):
        render_gif.main()


def test_fixed_contact_frames_match_mujoco_forward_kinematics(tmp_path):
    model, meta = bundle(tmp_path)
    # Explicit tree: root, then each hinge followed by its fixed contact frame.
    kin = Kinematics({
        **meta, 'kin_parent': [-1, 0, 1, 0, 3, 0, 5, 0, 7],
        'kin_joint_idx': [-1, 0, -1, 1, -1, 2, -1, 3, -1],
        'kin_axes': [[0, 1, 0]] * 9,
        'kin_off_p': [[0, 0, 0], [.2, .2, 0], [0, 0, -.2], [.2, -.2, 0], [0, 0, -.2],
                      [-.2, .2, 0], [0, 0, -.2], [-.2, -.2, 0], [0, 0, -.2]],
        'kin_off_R': [np.eye(3).tolist()] * 9, 'kin_tracked_idx': [2, 4, 6, 8],
    })
    data = mujoco.MjData(model)
    for q in (np.zeros(4), np.array([.3, -.4, .7, -.2])):
        data.qpos[7:] = q
        mujoco.mj_forward(model, data)
        positions, rotations, _, _ = kin.forward(
            jnp.asarray(data.qpos[:3]), jnp.eye(3), jnp.zeros(3), jnp.zeros(3),
            jnp.asarray(q), jnp.zeros(4),
        )
        np.testing.assert_allclose(positions, [data.body(f'foot{i}').xpos for i in range(4)], atol=1e-7)
        np.testing.assert_allclose(rotations, [data.body(f'foot{i}').xmat.reshape(3, 3) for i in range(4)], atol=1e-7)


def test_mirroring_is_declared_and_involutive_without_name_heuristics():
    layout = XiLayout({'joint_names': ['a', 'b', 'c', 'd'],
                       'dr': {'friction_range': [.3, 1.2], 'joint_offset': {'.*': .01}}})
    mirror = Mirror([1, 0, 3, 2], [-1, -1, 1, 1], layout)
    q = jnp.asarray([1., 2., 3., 4.])
    np.testing.assert_array_equal(mirror.joints(q), [-2., -1., 4., 3.])
    np.testing.assert_array_equal(mirror.joints(mirror.joints(q)), q)
    with pytest.raises(ValueError, match='involution'):
        Mirror([1, 2, 3, 0], [1, 1, 1, 1], layout)
    with pytest.raises(ValueError, match='involution'):
        Mirror([1, 0, 3, 2], [1, -1, 1, 1], layout)


def test_published_g1_mirror_mapping_is_unchanged():
    meta = load_meta()
    mirror = Mirror(meta['mirror_permutation'], meta['mirror_signs'], XiLayout(meta))
    for i, name in enumerate(meta['joint_names']):
        partner = name.replace('left_', '@').replace('right_', 'left_').replace('@', 'right_')
        assert meta['joint_names'][mirror.perm[i]] == partner
        assert mirror.sign[i] == (-1 if any(f'_{kind}_' in name for kind in ('roll', 'yaw')) else 1)
