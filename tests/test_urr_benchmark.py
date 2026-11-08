"""Independent planar geometry and actual MuJoCo state/custody checks."""
from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sx_actions.urr import MotionFrame

from test_retarget_target import fixture_body
from xmimic.urr_benchmark import TaskConfig, URRPhysicsEnv, digest, evaluate


def prepared_fixture(tmp_path, monkeypatch):
    body = fixture_body(tmp_path, monkeypatch)
    output = tmp_path / 'prepared'
    (output / 'robot').mkdir(parents=True)
    (output / 'embodiment.json').write_text(body.to_json())
    # Same independent two-link geometry as the canonical fixture; declared
    # nominal inertias/servos supply physics rather than a kinematic step stub.
    scene = output / 'robot/scene.xml'
    scene.write_text('''<mujoco><compiler angle="radian"/><option gravity="0 0 0" timestep=".002"/>
      <worldbody><body name="base"><body name="a"><joint name="q1" axis="0 0 1" range="-2 2" damping="1"/>
      <geom type="capsule" fromto="0 0 0 1 0 0" size=".02" mass="1" contype="0" conaffinity="0"/>
      <body name="b" pos="1 0 0"><joint name="q2" axis="0 0 1" range="-2 2" damping="1"/>
      <geom type="capsule" fromto="0 0 0 1 0 0" size=".02" mass="1" contype="0" conaffinity="0"/>
      <body name="tip" pos="1 0 0"/></body></body>
      <body name="held"><joint name="q3" axis="0 0 1" range="-2 2" damping="1" armature="0.1"/>
      <geom type="sphere" size=".02" mass=".1" contype="0" conaffinity="0"/></body></body></worldbody>
      <actuator><position joint="q1" kp="30" kv="6"/><position joint="q2" kp="30" kv="6"/>
      <position joint="q3" kp="30" kv="6"/></actuator></mujoco>''')
    spec = {'embodiment_id': str(body.id), 'joint_names': list(body.joint_names),
            'nominal_q': [.4, .3, 0.], 'action_scale': [.1]*3, 'control_dt': .02,
            'simulation_substeps': 10, 'roles': {'hand': 'tip_site'}, 'apertures': {},
            'assets': [], 'scene_sha256': digest(scene), 'urdf_sha256': str(body.urdf.asset.content.sha256)}
    (output / 'benchmark.json').write_text(json.dumps(spec))
    return output


def test_targets_match_independent_physics_and_partition_is_distinct(tmp_path, monkeypatch):
    prepared = prepared_fixture(tmp_path, monkeypatch)
    env = URRPhysicsEnv(prepared)
    env.reset(9)
    target = env.target_q.copy()
    env._set_positions(env.data, target)
    np.testing.assert_allclose(env.errors(), 0, atol=1e-14)
    evaluation = URRPhysicsEnv(prepared, TaskConfig(partition='evaluation'))
    evaluation.reset(9)
    assert not np.array_equal(target, evaluation.target_q)
    with pytest.raises(ValueError, match='exactly'):
        env.set_goal(MotionFrame(0, 'world', {}, {'invented_gripper': 0.}))


def test_complete_physics_and_goal_resume_across_episode_lengths(tmp_path, monkeypatch):
    prepared = prepared_fixture(tmp_path, monkeypatch)
    cfg = TaskConfig(precision=True, accumulate=True)
    env = URRPhysicsEnv(prepared, cfg)
    env.reset(0)
    state_shape = {k:v.shape for k,v in env.state().items()}
    for i in range(13):
        env.step(np.array([.1, -.07, .2]))
    saved = env.state()
    expected = env.step(np.array([-.2, .1, .0]))
    continued = URRPhysicsEnv(prepared, cfg)
    continued.reset(999)
    continued.restore(saved)
    actual = continued.step(np.array([-.2, .1, .0]))
    np.testing.assert_array_equal(expected[0], actual[0])
    assert expected[1:] == actual[1:]
    assert actual[3]['physics_warning_count'] == 0
    for k,v in continued.state().items():
        np.testing.assert_array_equal(v, env.state()[k])
    continued.reset(74828)
    assert state_shape == {k:v.shape for k,v in continued.state().items()}


def test_precision_reward_uses_vector_tolerance_and_exact_action_history(tmp_path, monkeypatch):
    prepared = prepared_fixture(tmp_path, monkeypatch)
    env = URRPhysicsEnv(prepared, TaskConfig(precision=True))
    obs = env.reset(0)
    error = np.zeros(env.error_dim)
    error[:2] = .0008
    assert float(env._ratio(error)) > 1  # each axis alone is inside; the vector is not.
    obs[env.error_slice] = error
    obs[3*env.action_dim:4*env.action_dim] = .5
    before = obs.copy()
    before[3*env.action_dim:4*env.action_dim] = 0.
    action = np.full(env.action_dim, .5)
    batch = env.reward_transition(before, jnp.stack([jnp.asarray(obs), jnp.asarray(obs)]), action)
    scalar = env.reward_transition(before, obs, action)
    np.testing.assert_allclose(batch, [scalar, scalar], atol=1e-7)
    assert float(env.reward_transition(obs, obs, action)) > float(scalar)


def test_evaluation_preserves_all_URR_samples_and_refuses_policy_rebinding(tmp_path, monkeypatch):
    prepared = prepared_fixture(tmp_path, monkeypatch)
    cfg = TaskConfig(episode_steps=10, hold_steps=3, partition='evaluation', accumulate=True)
    def factory():
        return URRPhysicsEnv(prepared, cfg)
    def zero_policy(obs):
        return np.zeros(3)
    output = tmp_path / 'evaluation'
    first = evaluate(factory, zero_policy, [3, 4], output, policy_identity='checkpoint-A')
    assert first == evaluate(factory, zero_policy, [3, 4], output, policy_identity='checkpoint-A')
    task = json.loads((output/'task-3.json').read_bytes())
    assert len(task['achieved']['frames']) == len(task['targets']['frames']) == len(task['native_q']) == 10
    assert not first['training_admitted']
    assert 'errors_terminal_window' in first
    with pytest.raises(ValueError, match='another immutable'):
        evaluate(factory, zero_policy, [3, 4], output, policy_identity='checkpoint-B')
    with pytest.raises(ValueError, match='immutable policy'):
        evaluate(factory, zero_policy, [3, 4], tmp_path/'missing')
    with pytest.raises(ValueError, match='disjoint'):
        evaluate(lambda: URRPhysicsEnv(prepared, replace(cfg, partition='train')), None, [1], tmp_path/'train')


def test_returned_pose_matches_current_integrated_native_state(tmp_path, monkeypatch):
    env = URRPhysicsEnv(prepared_fixture(tmp_path, monkeypatch), TaskConfig(accumulate=True))
    env.reset(4)
    _, _, _, metrics = env.step(np.array([.5, -.3, .2]))
    assert metrics['physics_warning_count'] == 0
    integrated_goal = env._goal_for(env.data.qpos[env.qadr].copy())
    env.set_goal(integrated_goal)
    np.testing.assert_allclose(env.errors(), 0, atol=1e-14)


def test_precision_reward_has_finite_zero_subgradient(tmp_path, monkeypatch):
    env = URRPhysicsEnv(prepared_fixture(tmp_path, monkeypatch), TaskConfig(precision=True))
    obs = env.reset(0)
    obs[env.error_slice] = 0
    gradient = jax.grad(lambda x: env.reward_transition(obs, x, jnp.zeros(env.action_dim)))(jnp.asarray(obs))
    assert np.isfinite(gradient).all()
    obs[env.error_slice.start] = .003
    gradient = jax.grad(lambda x: env.reward_transition(obs, x, jnp.zeros(env.action_dim)))(jnp.asarray(obs))
    assert np.isfinite(gradient).all()  # exact orientation with nonzero position


def test_copied_task_cannot_be_reused_for_another_request(tmp_path, monkeypatch):
    prepared = prepared_fixture(tmp_path, monkeypatch)
    cfg = TaskConfig(episode_steps=3, hold_steps=1, partition='evaluation', accumulate=True)
    def factory():
        return URRPhysicsEnv(prepared, cfg)
    def zero(obs):
        return np.zeros(3)
    old, new = tmp_path/'old', tmp_path/'new'
    evaluate(factory, zero, [1], old, policy_identity='A')
    new.mkdir()
    (new/'task-1.json').write_bytes((old/'task-1.json').read_bytes())
    with pytest.raises(ValueError, match='another evaluation'):
        evaluate(factory, zero, [1], new, policy_identity='B')


@pytest.mark.parametrize('precision,accumulate', [(False, False), (False, True), (True, False), (True, True)])
def test_structured_observation_matches_physics_and_ignores_hallucinated_channels(tmp_path, monkeypatch, precision, accumulate):
    env = URRPhysicsEnv(prepared_fixture(tmp_path, monkeypatch), TaskConfig(precision=precision, accumulate=accumulate))
    previous = env.reset(18)
    action = np.array([.15, -.21, .12])
    following, reward, _, metrics = env.step(action)
    assert metrics['physics_warning_count'] == 0
    proposal = following.copy()
    proposal[2*env.action_dim:] = 99.  # forbidden invented goals/errors/command/history
    projected = env.project_observation(previous, action, proposal)
    np.testing.assert_allclose(projected, following, atol=3e-4, rtol=1e-5)
    np.testing.assert_array_equal(projected[env.goal_slice], previous[env.goal_slice])
    np.testing.assert_allclose(env.reward_transition(previous, projected, action), reward, atol=2e-5)
    jacobian = jax.jacrev(lambda prediction: env.project_observation(previous, action, prediction))(jnp.asarray(proposal))
    assert np.isfinite(jacobian).all()
    np.testing.assert_array_equal(jacobian[:, 2*env.action_dim:], 0.)


def test_structured_identity_goal_gradient_is_finite_and_moving_state_is_refused(tmp_path, monkeypatch):
    prepared = prepared_fixture(tmp_path, monkeypatch)
    env = URRPhysicsEnv(prepared, TaskConfig(precision=True, accumulate=True))
    env.reset(0)
    env.set_goal(env._goal_for(env.data.qpos[env.qadr]))
    obs = env.observation()
    grad = jax.grad(lambda predicted: env.reward_transition(
        obs, env.project_observation(obs, jnp.zeros(env.action_dim), predicted), jnp.zeros(env.action_dim)))(jnp.asarray(obs))
    assert np.isfinite(grad).all()
    moving = URRPhysicsEnv(prepared, TaskConfig(moving=True, episode_steps=6, hold_steps=2))
    initial = moving.reset(0)
    np.testing.assert_allclose(moving.errors(), 0., atol=1e-14)
    with pytest.raises(ValueError, match='stationary goals'):
        moving.project_observation(initial, np.zeros(moving.action_dim), initial)
    for _ in range(4):
        moving.step(np.zeros(moving.action_dim))
    final_goal = moving._goal_for(moving.target_q)
    for role in moving.roles:
        np.testing.assert_allclose(moving.goal.poses[role].position, final_goal.poses[role].position, atol=1e-14)
    assert moving.progress == 0.  # changing targets cannot create improvement rewards


@pytest.mark.parametrize('curved', [False, True])
def test_structured_aperture_uses_declared_owner_relation(tmp_path, monkeypatch, curved):
    from sx_embodiments import Embodiment
    from sx_embodiments.compose import RootMount, body_component
    from sx_embodiments.curves import Curve1D, Knot
    from sx_embodiments.identity import PartId
    from sx_embodiments.layout import JointLayout
    from sx_embodiments.parts import GripperSpec, Limb

    prepared = prepared_fixture(tmp_path, monkeypatch)
    body = Embodiment.from_json((prepared/'embodiment.json').read_text())
    arm = body.components[0].part
    gripper = GripperSpec(PartId('declared_aperture'), JointLayout((arm.layout.axes[2],)),
        travel_m=(0., .07), gap_curve=Curve1D((Knot(-2., 0.), Knot(.3, .01), Knot(2., .07))) if curved else None)
    body = replace(body, components=(
        body_component('arm', replace(arm, layout=JointLayout(arm.layout.axes[:2]), home=arm.home[:2]), RootMount('base'), limb=Limb.ARM),
        body_component('tool', gripper, RootMount('base'))))
    (prepared/'embodiment.json').write_text(body.to_json())
    spec = json.loads((prepared/'benchmark.json').read_text())
    spec['embodiment_id'] = str(body.id)
    spec['apertures'] = {'opening': {'component_id': 'tool', 'drive_joint': 'q3'}}
    (prepared/'benchmark.json').write_text(json.dumps(spec))
    env = URRPhysicsEnv(prepared)
    previous = env.reset(13)
    for value in (-2., -.5, .3, 1.4, 2.):
        q = env.data.qpos[env.qadr].copy()
        q[2] = value
        env._set_positions(env.data, q)
        proposed = env.observation()
        projected = env.project_observation(previous, np.zeros(3), proposed)
        expected = env.goal.apertures['opening']-gripper.aperture_from_drive(value)
        np.testing.assert_allclose(projected[env.error_slice][-1], expected, atol=1e-8)

        # The aperture reward is the declared 1D position-pair extension, not
        # fabricated physical keypoints or a normalized-joint proxy.
        actual_reward = env.reward_transition(previous, projected, np.zeros(3))
        zero_gap = np.asarray(projected).copy()
        zero_gap[env.error_slice.stop-1] = 0.
        delta = actual_reward-env.reward_transition(previous, zero_gap, np.zeros(3))
        np.testing.assert_allclose(delta, env.control_dt*(-.2*abs(expected)-.1*np.tanh(abs(expected)/.1)), atol=1e-8)
        error = np.zeros(env.error_dim)
        error[-1] = .00931
        np.testing.assert_allclose(env._precision_gain(error, np), 1.)

    # At an exact aperture target, JAX abs(0) chooses +1, which would push
    # model-based optimization away from the goal. Require the zero subgradient
    # for both tracking and global precision shaping, and descent on either side.
    exact = env.observation()
    exact[env.error_slice] = 0.
    exact[-4:] = [1., 0., 0., 0.]
    for precision in (False, True):
        env.config = replace(env.config, precision=precision)
        def reward(error):
            return env.reward_transition(exact, jnp.asarray(exact).at[env.error_slice].set(error), jnp.zeros(3))
        gradient = jax.grad(reward)
        np.testing.assert_array_equal(gradient(jnp.zeros(env.error_dim)), 0.)
        for sign in (-1., 1.):
            error = jnp.zeros(env.error_dim).at[-1].set(sign*.0003)
            assert sign*gradient(error)[-1] < 0.
