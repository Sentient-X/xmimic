"""Actual CPU MuJoCo transitions, learned updates and exact retained continuation."""
from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from xmimic import checkpoint
from xmimic.urr_learning import LearningConfig, _initial, load_policy, predict_next, train


class ServoEnvironment:
    """One physical hinged mechanism, using the production flat observation contract."""
    observation_dim = 6
    action_dim = 1
    binding = {'fixture': 'mujoco-hinge-v1', 'precision': False, 'accumulate': True}

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string('''<mujoco>
          <option timestep="0.005" gravity="0 0 0"/>
          <worldbody><body><joint name="j" type="hinge" axis="0 0 1" damping="0.3" range="-90 90"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.2"/></body></worldbody>
          <actuator><position joint="j" kp="4"/></actuator></mujoco>''')
        self.data = mujoco.MjData(self.model)

    def reset(self, seed):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = rng.uniform(-.3, .3)
        self.command = float(self.data.qpos[0]+rng.uniform(-.1, .1))
        self.goal = float(rng.uniform(-.5, .5))
        self.previous = 0.
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def observation(self):
        q = self.data.qpos[0]
        return np.array([q, .1*self.data.qvel[0], self.command-q, self.previous, self.goal, self.goal-q], np.float32)

    def step(self, action):
        previous = self.observation()
        self.previous = float(action[0])
        self.command = float(np.clip(self.command+.05*self.previous, -.9, .9))
        self.data.ctrl[0] = self.command
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        obs = self.observation()
        reward = float(self.reward_transition(previous, obs, action))
        return obs, reward, self.steps == 11, {}

    def reward_transition(self, previous, obs, action):
        return jnp.exp(-4*obs[..., 5]**2)-.001*jnp.sum((action-previous[..., 3:4])**2, axis=-1)

    def project_observation(self, previous, action, predicted):
        # Exact facts for this one-hinge physics fixture, in the same contract as
        # the canonical environment's FK-based observation projection.
        command = jnp.clip(previous[..., 0]+previous[..., 2]+.05*action[..., 0], -.9, .9)
        q, qd, goal = predicted[..., 0], predicted[..., 1], previous[..., 4]
        return jnp.stack((q, qd, command-q, action[..., 0], goal, goal-q), axis=-1)

    def state(self):
        return {'qpos': self.data.qpos.copy(), 'qvel': self.data.qvel.copy(), 'ctrl': self.data.ctrl.copy(),
                'time': np.array(self.data.time), 'qacc_warmstart': self.data.qacc_warmstart.copy(),
                'command': np.array(self.command), 'goal': np.array(self.goal),
                'previous': np.array(self.previous), 'steps': np.array(self.steps, np.int64)}

    def restore(self, state):
        for name in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart'):
            getattr(self.data, name)[:] = state[name]
        self.data.time = float(state['time'])
        self.command, self.goal, self.previous = (float(state[k]) for k in ('command', 'goal', 'previous'))
        self.steps = int(state['steps'])
        mujoco.mj_forward(self.model, self.data)


def restored(path):
    binding = json.loads((path/'manifest.json').read_text())['binding']
    cfg = LearningConfig(**binding['config'])
    state = {k:np.zeros(v['shape'], dtype=v['dtype']) for k,v in binding['environment_state'].items()}
    template = _initial(cfg, binding['observation_dim'], binding['action_dim'], state)
    return checkpoint.restore_checkpoint(path, {}, template, binding=binding)[1]


@pytest.mark.parametrize('method', ['ppo', 'world_model'])
def test_real_physics_learning_and_interruption_resume_are_identical(tmp_path, method):
    cfg = LearningConfig(iterations=3, rollout_steps=8, hidden=(12,), ppo_epochs=2,
                         wm_updates=2, batch_size=8, imagination_horizon=2, replay_capacity=32)
    whole = train(ServoEnvironment, method=method, precision=False, config=cfg, output=tmp_path/'whole')
    calls = 0
    def stop():
        nonlocal calls
        calls += 1
        return calls == 3
    interrupted = train(ServoEnvironment, method=method, precision=False, config=cfg,
                        output=tmp_path/'resumed', stop_requested=stop)
    assert interrupted.receipt['iterations'] == 2 and not interrupted.receipt['completed']
    resumed = train(ServoEnvironment, method=method, precision=False, config=cfg,
                    output=tmp_path/'resumed', resume_from=interrupted.checkpoint)
    a, b = restored(whole.checkpoint), restored(resumed.checkpoint)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)
    assert whole.receipt['metrics'] == resumed.receipt['metrics']
    assert resumed.receipt['physics_steps'] == 24
    initial = _initial(cfg, 6, 1, a['env'])
    assert any(not np.array_equal(x, y) for x, y in zip(jax.tree.leaves(initial['pi']), jax.tree.leaves(a['pi']), strict=True))
    assert np.max(np.abs(a['replay_next'][:24, 0]-a['replay_obs'][:24, 0])) > 1e-4
    policy = load_policy(resumed.checkpoint)
    obs = ServoEnvironment().reset(59)
    action = policy(obs)
    assert action.shape == (1,) and np.isfinite(action).all() and np.max(np.abs(action)) <= 1
    different_command = obs.copy()
    different_command[2] += .5
    assert not np.array_equal(action, policy(different_command))
    with pytest.raises(ValueError, match='binding differs'):
        train(ServoEnvironment, method=method, precision=False, config=replace(cfg, gamma=.9),
              output=tmp_path/'wrong', resume_from=resumed.checkpoint)


def test_world_model_fits_measured_transition_and_retains_command_context(tmp_path):
    cfg = LearningConfig(iterations=4, rollout_steps=32, hidden=(24,), wm_updates=30,
                         batch_size=32, imagination_horizon=2, replay_capacity=256, learning_rate=.001)
    result = train(ServoEnvironment, method='world_model', precision=False, config=cfg, output=tmp_path)
    state = restored(result.checkpoint)
    # Separate seeds generate held-out physics transitions, never optimizer labels.
    env = ServoEnvironment()
    obs, actions, following = [], [], []
    rng = np.random.default_rng(418)
    for seed in (301, 302, 303):
        current = env.reset(seed)
        for _ in range(10):
            action = rng.uniform(-.7, .7, 1).astype(np.float32)
            nxt, _, _, _ = env.step(action)
            obs.append(current)
            actions.append(action)
            following.append(nxt)
            current = nxt
    obs, actions, following = map(jnp.asarray, (np.array(obs), np.array(actions), np.array(following)))
    initial = _initial(cfg, 6, 1, state['env'])
    baseline = predict_next(initial['wm'], state['wm_buf'], obs, actions)
    prediction = predict_next(state['wm'], state['wm_buf'], obs, actions)
    # Physical q/velocity and accumulated command, excluding the exact previous-action slot.
    baseline_error = float(jnp.mean((baseline[:, :3]-following[:, :3])**2))
    learned_error = float(jnp.mean((prediction[:, :3]-following[:, :3])**2))
    assert learned_error < baseline_error
    response = jax.jacrev(lambda o: predict_next(state['wm'], state['wm_buf'], o, jnp.zeros(1)))(obs[0])
    assert np.linalg.norm(np.asarray(response[:2, 2])) > 1e-5
    np.testing.assert_array_equal(prediction[:, 3:4], actions)
    # Equal measured q/velocity with a different retained command really has
    # different future physics; omitting this observation would alias the states.
    env.reset(99)
    first = env.state()
    first['command'] = np.array(.5)
    env.restore(first)
    with_command, _, _, _ = env.step(np.zeros(1))
    first['command'] = np.array(-.5)
    env.restore(first)
    opposite, _, _, _ = env.step(np.zeros(1))
    assert abs(with_command[0]-opposite[0]) > .01
