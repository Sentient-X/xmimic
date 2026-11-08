"""Independent scalar physics parity and exact batched learning continuation."""
from dataclasses import replace
import json

import jax
import numpy as np
import pytest

from test_urr_benchmark import prepared_fixture
from xmimic import checkpoint
from xmimic.urr_benchmark import TaskConfig, URRPhysicsEnv
from xmimic.urr_learning import LearningConfig, _initial, load_policy, train


@pytest.mark.parametrize('precision,accumulate', [(False, False), (False, True), (True, False), (True, True)])
def test_batch_matches_independent_scalar_physics_and_full_resume(tmp_path, monkeypatch, precision, accumulate):
    prepared = prepared_fixture(tmp_path, monkeypatch)
    cfg = TaskConfig(precision=precision, accumulate=accumulate, episode_steps=12, hold_steps=3)
    template = URRPhysicsEnv(prepared, cfg)
    batch = template.vectorize(4)
    resumed = template.vectorize(4)
    scalars = [URRPhysicsEnv(prepared, cfg) for _ in range(4)]
    try:
        initial = batch.reset(np.arange(4)+712)
        expected = np.stack([env.reset(712+i) for i, env in enumerate(scalars)])
        np.testing.assert_array_equal(initial, expected)
        assert all(env.model is template.model for env in batch.envs)
        assert len({id(env.data) for env in batch.envs}) == 4
        for step in range(9):
            action = np.random.default_rng(911+step).uniform(-.4, .4, (4, template.action_dim))
            actual, reward, done, info = batch.step(action)
            independently = [env.step(a) for env, a in zip(scalars, action, strict=True)]
            np.testing.assert_allclose(actual, np.stack([x[0] for x in independently]), atol=4e-5, rtol=1e-5)
            np.testing.assert_allclose(reward, [x[1] for x in independently], atol=2e-6)
            np.testing.assert_array_equal(done, [x[2] for x in independently])
            assert not info['physics_warning_count'].any()
            for env, scalar in zip(batch.envs, scalars, strict=True):
                np.testing.assert_allclose(env.data.qpos, scalar.data.qpos, atol=2e-8, rtol=1e-6)
                np.testing.assert_allclose(env.data.qvel, scalar.data.qvel, atol=2e-7, rtol=1e-6)
            if step == 4:
                resumed.restore(batch.state())
            elif step > 4:
                repeat = resumed.step(action)
                np.testing.assert_array_equal(repeat[0], actual)
                np.testing.assert_array_equal(repeat[1], reward)
                for key, state in batch.state().items():
                    np.testing.assert_array_equal(resumed.state()[key], state)
        state = batch.state()
        batch.reset_at([1], [812])
        for key, value in state.items():
            np.testing.assert_array_equal(batch.state()[key][[0, 2, 3]], value[[0, 2, 3]])
        assert not np.array_equal(batch.state()['target_q'][1], state['target_q'][1])
    finally:
        batch.close()
        resumed.close()


def restore_learning(path):
    binding = json.loads((path/'manifest.json').read_text())['binding']
    config = LearningConfig(**binding['config'])
    state = {key: np.zeros(value['shape'], dtype=value['dtype']) for key, value in binding['environment_state'].items()}
    return checkpoint.restore_checkpoint(path, {}, _initial(config, binding['observation_dim'], binding['action_dim'], state), binding=binding)[1]


@pytest.mark.parametrize('method', ['ppo', 'world_model'])
@pytest.mark.parametrize('precision', [False, True])
def test_batched_real_learning_replay_and_checkpoint_resume(tmp_path, monkeypatch, method, precision):
    prepared = prepared_fixture(tmp_path, monkeypatch)
    def factory():
        return URRPhysicsEnv(prepared, TaskConfig(episode_steps=3, hold_steps=1, accumulate=True, precision=precision))
    config = LearningConfig(seed=571, num_envs=3, iterations=3, rollout_steps=4, hidden=(8,),
                            ppo_epochs=1, wm_updates=1, batch_size=4, imagination_horizon=2, replay_capacity=16)
    whole = train(factory, method=method, precision=precision, config=config, output=tmp_path/'whole')
    calls = 0
    def stop():
        nonlocal calls
        calls += 1
        return calls == 3
    interrupted = train(factory, method=method, precision=precision, config=config, output=tmp_path/'continue', stop_requested=stop)
    continued = train(factory, method=method, precision=precision, config=config, output=tmp_path/'continue', resume_from=interrupted.checkpoint)
    left, right = restore_learning(whole.checkpoint), restore_learning(continued.checkpoint)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)
    assert continued.receipt['physics_steps'] == 36
    np.testing.assert_array_equal(right['resets'], [4, 4, 4])
    assert right['replay_size'] == 16 and right['replay_cursor'] == 4
    assert right['env']['physics'].shape[0] == 3
    assert not np.array_equal(right['replay_obs'][0], right['replay_obs'][1])
    policy = load_policy(continued.checkpoint)
    assert policy(factory().reset(671)).shape == (3,)
    with pytest.raises(ValueError, match='binding differs'):
        train(factory, method=method, precision=precision, config=replace(config, num_envs=2), output=tmp_path/'wrong', resume_from=continued.checkpoint)


def test_batch_refuses_unrepresented_moving_state(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='stationary'):
        URRPhysicsEnv(prepared_fixture(tmp_path, monkeypatch), TaskConfig(moving=True)).vectorize(4)


def test_gae_terminal_masks_do_not_cross_environment_lanes():
    from xmimic.urr_learning import _advantages
    config = LearningConfig(num_envs=2, rollout_steps=3, gamma=1., gae_lambda=1.)
    rewards = np.array([[1., 2.], [3., 4.], [5., 6.]], np.float32)
    values = np.array([[.1, .2], [.3, .4], [.5, .6]], np.float32)
    following = np.array([[.3, .4], [999., .6], [.7, 999.]], np.float32)
    done = np.array([[False, False], [True, False], [False, True]])
    advantage, returns = _advantages(rewards, done, values, following, config)
    np.testing.assert_allclose(returns, [[4., 12.], [3., 10.], [5.7, 6.]], atol=1e-6)
    np.testing.assert_allclose(advantage+values, returns)


@pytest.mark.parametrize('num_envs', [1, 4])
def test_learning_refuses_evaluation_partition(tmp_path, monkeypatch, num_envs):
    prepared = prepared_fixture(tmp_path, monkeypatch)
    def factory():
        return URRPhysicsEnv(prepared, TaskConfig(partition='evaluation'))
    with pytest.raises(ValueError, match='Evaluation task partition'):
        train(factory, method='ppo', precision=False, config=LearningConfig(num_envs=num_envs), output=tmp_path/'no-training')
    assert not (tmp_path/'no-training').exists()
