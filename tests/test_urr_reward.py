"""Independent numeric oracles for the pinned source-term URR adaptation.

PolicyIK 22f3a5c tasks/mdp.py and env_cfg.py
    inherited OpenArm bad82e23
reach_env_cfg.py/rewards.py. Oracle uses explicit SciPy-transformed keypoints,
not production rotation reconstruction or scalar coordinate kernels.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_urr_benchmark import prepared_fixture
from xmimic.urr_benchmark import TaskConfig, URRPhysicsEnv, policyik_pose_reward
from xmimic.urr_learning import LearningConfig, train


def source_pose(position, actual, goal):
    offsets = .45*np.eye(3)
    offsets -= offsets.mean(0)
    distances = [np.linalg.norm(position+goal@point-actual@point) for point in offsets]
    d, k = np.linalg.norm(position), np.mean(distances)
    return -.2*d+.1*(1-np.tanh(d/.1))+1-np.tanh(k/.2)+1-np.tanh(k/.05)


@pytest.mark.parametrize('seed', [0, 12, 184])
def test_literal_source_keypoints_and_common_rigid_frame_invariance(seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(-.2, .2, 3)
    actual, goal, frame = Rotation.random(3, random_state=rng).as_matrix()
    error = Rotation.from_matrix(goal@actual.T).as_rotvec()
    expected = source_pose(p, actual, goal)
    value = policyik_pose_reward(p, error, goal)
    np.testing.assert_allclose(value, expected, atol=3e-7)
    # A common rigid-frame translation cancels in position error. Its rotation
    # transforms both errors and rotations, without altering physical mismatch.
    moved = policyik_pose_reward(frame@p, frame@error, frame@goal)
    np.testing.assert_allclose(moved, value, atol=4e-7)


def test_old_axis_sacrifice_counterexample_and_far_field_descent():
    a, b = jnp.array([.02, .02, 0.]), jnp.array([.035, 0., 0.])
    assert jnp.exp(-(b/.01)**2).mean() > jnp.exp(-(a/.01)**2).mean()
    def reward(p):
        return policyik_pose_reward(p, jnp.zeros(3), jnp.eye(3))
    assert reward(b) < reward(a)  # worse Euclidean pose is now worse tracking
    for distance in (.05, .1, 1.):
        gradient = jax.grad(reward)(jnp.array([distance, 0., 0.]))
        assert gradient[0] <= -.199999 and np.isfinite(gradient).all()
    assert np.isfinite(jax.grad(reward)(jnp.zeros(3))).all()
    assert np.isfinite(jax.grad(lambda r: policyik_pose_reward(jnp.zeros(3), r, jnp.eye(3)))(jnp.zeros(3))).all()


def observation_with_error(env, base, error, *, best, valid=True):
    obs = base.copy()
    obs[env.error_slice] = error
    obs[:env.action_dim] = 0.  # centred native joints, so no margin cost
    obs[3*env.action_dim:4*env.action_dim] = 0.
    obs[-4:] = [valid, best/100., 0., 0.]
    return obs


def test_single_pose_no_aperture_matches_source_terms_and_dt(tmp_path, monkeypatch):
    env = URRPhysicsEnv(prepared_fixture(tmp_path, monkeypatch))
    previous = env.reset(3)
    error = np.array([.07, -.02, .03, .4, -.1, .2])
    following = observation_with_error(env, previous, error, best=100.)
    goal = following[env.goal_slice]
    R = np.column_stack((goal[3:6], goal[6:9], np.cross(goal[3:6], goal[6:9])))
    actual = Rotation.from_rotvec(-error[3:]).as_matrix()@R
    expected = env.control_dt*source_pose(error[:3], actual, R)
    np.testing.assert_allclose(env.reward_transition(previous, following, np.zeros(3)), expected, atol=1e-8)


def test_exact_precision_history_reset_binary_dwell_and_potential(tmp_path, monkeypatch):
    env = URRPhysicsEnv(prepared_fixture(tmp_path, monkeypatch), TaskConfig(precision=True))
    base = env.reset(7)
    zero = np.zeros(env.action_dim)
    def obs(rho, best, valid=True):
        return observation_with_error(env, base, [rho*.001, 0, 0, 0, 0, 0], best=best, valid=valid)
    previous, following = obs(2., 1.8), obs(.8, .8)
    def tracking(o):
        old = env.config
        env.config = replace(old, precision=False)
        try:
            return float(env.reward_transition(previous, o, zero))
        finally:
            env.config = old
    def u(x):
        return .5*(1+np.tanh((1-x)/.5))
    shaping = 26.831*(u(.8)-u(1.8)) + (-.99*.8+2.) + 1.
    np.testing.assert_allclose(env.reward_transition(previous, following, zero)-tracking(following), .02*shaping, atol=2e-7)
    # Dwell is binary per successful sample, not an accumulating reward ramp.
    longer = following.copy()
    longer[-1] = 100.
    np.testing.assert_array_equal(env.reward_transition(previous, longer, zero), env.reward_transition(previous, following, zero))
    first = previous.copy()
    first[-4] = 0.
    np.testing.assert_allclose(env.reward_transition(first, following, zero), tracking(following), atol=1e-8)
    changed = following.copy()
    changed[env.goal_slice.start] += .1
    reset_previous = previous.copy()
    reset_previous[-4] = 0.
    np.testing.assert_allclose(env.reward_transition(previous, changed, zero), env.reward_transition(reset_previous, changed, zero), atol=1e-8)
    # Cap, regression penalty and positive discounted potential at equal rho
    # are literal source semantics, not a newly optimized reward variant.
    previous, following = obs(100., 1.), obs(101., 1.)
    expected = .02*(.5-.2*3.)
    np.testing.assert_allclose(env.reward_transition(previous, following, zero)-tracking(following), expected, atol=1e-7)


def test_history_projection_first_transition_and_native_margin(tmp_path, monkeypatch):
    env = URRPhysicsEnv(prepared_fixture(tmp_path, monkeypatch), TaskConfig(precision=True, accumulate=True))
    initial = env.reset(1)
    assert initial[-4] == 0.
    following, reward, _, _ = env.step(np.zeros(3))
    assert following[-4] == 1.
    np.testing.assert_allclose(following[-3]*100, env._ratio(following[env.error_slice]), rtol=1e-6)
    assert following[-2] == 0.
    projected = env.project_observation(initial, np.zeros(3), following)
    np.testing.assert_allclose(env.reward_transition(initial, projected, np.zeros(3)), reward, atol=2e-6)
    centre = observation_with_error(env, initial, [.02, 0, 0, 0, 0, 0], best=20.)
    edge = centre.copy()
    edge[0] = .96  # margin .02 of full native range
    delta = env.reward_transition(initial, edge, np.zeros(3))-env.reward_transition(initial, centre, np.zeros(3))
    np.testing.assert_allclose(delta, -.02*.6, atol=1e-8)
    gradient = jax.grad(lambda o: env.reward_transition(initial, o, jnp.zeros(3)))(jnp.asarray(centre))
    assert np.isfinite(gradient).all()


def test_source_gain_thresholds_and_discount_binding(tmp_path, monkeypatch):
    prepared = prepared_fixture(tmp_path, monkeypatch)
    env = URRPhysicsEnv(prepared, TaskConfig(precision=True, accumulate=True))
    error = np.zeros(6)
    assert env._precision_gain(error, np) == .1
    error[0] = .00931
    np.testing.assert_allclose(env._precision_gain(error, np), 1.)
    error[0] = (.00931+.0005)/2
    np.testing.assert_allclose(env._precision_gain(error, np), .55)
    error[:] = 0.
    error[3] = np.deg2rad(2.328)
    np.testing.assert_allclose(env._precision_gain(error, np), 1.)
    np.testing.assert_allclose(env._precision_gain(jnp.asarray(error), jnp), env._precision_gain(error, np), atol=1e-7)
    with pytest.raises(ValueError, match='discount'):
        train(lambda: env, method='ppo', precision=True, config=LearningConfig(gamma=.95), output=tmp_path/'refused')
