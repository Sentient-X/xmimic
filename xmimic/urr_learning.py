"""Matched stationary URR learners, explicitly adapted rather than vanilla baselines.

PolicyIK-derived: clipped PPO/GAE with the common environment's command feedback
and optional precision package (upstream 22f3a5c, rl_games_ppo_cfg.yaml).
XMimic-derived: fit a residual dynamics model to actual physics transitions, then
backpropagate task reward through imagined rollouts. Reuses XMimic nets/optim and
checkpoint custody; replaces its floating-root/q-reference MDP. These CPU pilots
omit upstream scale, mirroring, failure search and distributed execution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from . import checkpoint, nets, optim


class Environment(Protocol):
    observation_dim: int
    action_dim: int
    binding: dict

    def reset(self, seed: int) -> np.ndarray: ...
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]: ...
    def state(self) -> dict[str, np.ndarray]: ...
    def restore(self, state: dict[str, np.ndarray]) -> None: ...
    def reward_from_observation(self, observation, previous_action): ...
    def project_observation(self, previous, action, predicted): ...


@dataclass(frozen=True)
class LearningConfig:
    seed: int = 0
    iterations: int = 20
    rollout_steps: int = 128
    hidden: tuple[int, ...] = (64, 64)
    learning_rate: float = 3e-4
    gamma: float = .99
    gae_lambda: float = .95
    ppo_clip: float = .2
    entropy_weight: float = .01
    ppo_epochs: int = 4
    exploration_std: float = .3
    replay_capacity: int = 4096
    batch_size: int = 128
    wm_updates: int = 8
    imagination_horizon: int = 8
    gradient_clip: float = 1.
    num_envs: int = 1

    def __post_init__(self):
        object.__setattr__(self, 'hidden', tuple(self.hidden))
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a non-negative integer')
        for name in ('iterations', 'rollout_steps', 'ppo_epochs', 'replay_capacity', 'batch_size', 'wm_updates', 'imagination_horizon', 'num_envs'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if not self.hidden or any(type(x) is not int or x < 1 for x in self.hidden):
            raise ValueError('positive hidden widths required')
        for name in ('learning_rate', 'exploration_std', 'gradient_clip'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1 or not 0 < self.ppo_clip < 1:
            raise ValueError('invalid discount, GAE or PPO clip')
        if not np.isfinite(self.entropy_weight) or self.entropy_weight < 0:
            raise ValueError('invalid entropy weight')


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: Path
    receipt: dict


def _mask(params):
    return jax.tree.map(lambda _: False, params)


def _initial(config, observation_dim, action_dim, environment_state):
    keys = jax.random.split(jax.random.PRNGKey(config.seed), 4)
    pi = {'net': nets.mlp_init(keys[0], [observation_dim, *config.hidden, action_dim]),
          'log_std': jnp.full(action_dim, np.log(config.exploration_std))}
    pi['net'][-1]['w'] *= .01
    pi['net'][-1]['b'] *= 0
    value = {'net': nets.mlp_init(keys[1], [observation_dim, *config.hidden, 1])}
    wm, wm_buf = nets.wm_init(keys[2], observation_dim+action_dim, observation_dim, config.hidden)
    n = config.replay_capacity
    return dict(pi=pi, value=value, wm=wm, wm_buf=wm_buf,
                opt_pi=optim.init(pi, _mask(pi)), opt_value=optim.init(value, _mask(value)),
                opt_wm=optim.init(wm, _mask(wm)), obs_norm=nets.rn_init(observation_dim),
                key=keys[3], it=np.array(0, np.int64), resets=np.zeros(config.num_envs, np.int64) if config.num_envs > 1 else np.array(0, np.int64),
                physics_steps=np.array(0, np.int64), obs=np.zeros((config.num_envs, observation_dim) if config.num_envs > 1 else observation_dim, np.float32),
                env=environment_state, replay_obs=np.zeros((n, observation_dim), np.float32),
                replay_action=np.zeros((n, action_dim), np.float32),
                replay_next=np.zeros((n, observation_dim), np.float32),
                replay_reward=np.zeros(n, np.float32), replay_done=np.zeros(n, np.float32),
                metric_history=np.zeros((config.iterations, 5), np.float64),
                replay_size=np.array(0, np.int64), replay_cursor=np.array(0, np.int64))


def _mean(pi, norm, obs):
    return nets.mlp_apply(pi['net'], nets.rn_apply(norm, obs), jax.nn.elu)


def _value(value, norm, obs):
    return nets.mlp_apply(value['net'], nets.rn_apply(norm, obs), jax.nn.elu)[..., 0]


def _logprob(pi, mean, latent):
    log_std = jnp.clip(pi['log_std'], -5., 1.)
    return (-.5*((latent-mean)*jnp.exp(-log_std))**2-log_std-.5*np.log(2*np.pi)).sum(-1)


def predict_next(wm, buffers, observation, action):
    """Residual prediction conditioned on complete command-bearing observation."""
    inputs = jnp.concatenate((observation, action), axis=-1)
    delta = nets.mlp_apply(wm['net'], nets.rn_apply(buffers['in_norm'], inputs), jax.nn.silu)
    following = observation + delta*nets.rn_std(buffers['out_norm'], 1e-4)+buffers['out_norm']['mean']
    # This slot is the applied action by the common environment contract, not
    # unknown physics. Retain it exactly for the next policy and action-rate cost.
    dimension = action.shape[-1]
    return following.at[..., 3*dimension:4*dimension].set(action)


def _updated(params, state, loss, config):
    value, grads = jax.value_and_grad(loss)(params)
    grads, norm = optim.clip_by_global_norm(grads, config.gradient_clip)
    updated, opt = optim.step(params, grads, state, _mask(params), config.learning_rate)
    # A nonfinite numerical program fails before publishing a checkpoint.
    return updated, opt, value, norm


def _kernels(config, reward, action_dim, project=None):
    @jax.jit
    def ppo(pi, value, op, ov, norm, obs, latent, old_logp, advantage, returns):
        def loss(params):
            ratio = jnp.exp(_logprob(params, _mean(params, norm, obs), latent)-old_logp)
            clipped = jnp.clip(ratio, 1-config.ppo_clip, 1+config.ppo_clip)
            entropy = jnp.sum(jnp.clip(params['log_std'], -5., 1.)+.5*np.log(2*np.pi*np.e))
            return -jnp.minimum(ratio*advantage, clipped*advantage).mean()-config.entropy_weight*entropy
        pi, op, pl, _ = _updated(pi, op, loss, config)
        value, ov, vl, _ = _updated(value, ov, lambda v: .5*jnp.mean((_value(v, norm, obs)-returns)**2), config)
        return pi, value, op, ov, pl, vl

    @jax.jit
    def fit(wm, opt, buffers, obs, actions, following):
        physical = slice(0, 2*action_dim)
        scale = nets.rn_std(buffers['out_norm'], 1e-3)[physical]
        def loss(w):
            prediction = project(obs, actions, predict_next(w, buffers, obs, actions))
            return jnp.mean(((prediction[..., physical]-following[..., physical])/scale)**2)
        return _updated(wm, opt, loss, config)

    @jax.jit
    def imagined(pi, opt, wm, buffers, norm, starts):
        def loss(params):
            def step(obs, index):
                action = jnp.tanh(_mean(params, norm, obs))
                following = project(obs, action, predict_next(wm, buffers, obs, action))
                # The common environment owns task semantics, including action-rate
                # cost. Complete command and reward-history state remain in obs.
                r = jax.vmap(reward)(following, obs[:, 3*action_dim:4*action_dim])
                return following, config.gamma**index*r
            _, rewards = jax.lax.scan(step, starts, jnp.arange(config.imagination_horizon))
            return -rewards.sum(0).mean()
        return _updated(pi, opt, loss, config)
    return ppo, fit, imagined


def _advantages(rewards, dones, values, next_values, config):
    """Time recurrence is independent for each lane; flatten only afterward."""
    batched = config.num_envs > 1
    advantage = np.zeros((config.rollout_steps, config.num_envs) if batched else config.rollout_steps, np.float32)
    running = 0.
    for index in range(config.rollout_steps-1, -1, -1):
        alive = 1-np.asarray(dones[index], dtype=np.float32) if batched else 1-float(dones[index])
        delta = rewards[index]+config.gamma*alive*next_values[index]-values[index]
        running = delta+config.gamma*config.gae_lambda*alive*running
        advantage[index] = running
    return advantage, advantage+np.asarray(values, np.float32)


def _program():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), Path(nets.__file__), Path(optim.__file__), Path(checkpoint.__file__))}


def _binding(env, config, method, precision, state):
    return dict(schema='xmimic.matched-urr-learning/v1', method=method, precision=precision,
                config=asdict(config), observation_dim=env.observation_dim, action_dim=env.action_dim,
                environment=env.binding, environment_state={k:dict(shape=list(v.shape), dtype=str(v.dtype)) for k,v in state.items()},
                code=_program(),
                runtime=dict(jax=jax.__version__, numpy=np.__version__, backend=jax.default_backend(),
                             matrix_output=str(nets.MM_OUT), x64=bool(jax.config.jax_enable_x64)))


def _observation(value, size, num_envs=1):
    result = np.asarray(value, np.float32)
    if result.shape != ((num_envs, size) if num_envs > 1 else (size,)) or not np.isfinite(result).all():
        raise ValueError('environment returned an invalid observation')
    return result.copy()


def _train(env_factory: Callable[[], Environment], *, method: Literal['ppo', 'world_model'], precision: bool,
          config: LearningConfig, output: Path, resume_from: Path | None = None,
          stop_requested: Callable[[], bool] | None = None) -> TrainingResult:
    if method not in ('ppo', 'world_model') or type(precision) is not bool:
        raise ValueError('explicit matched method and precision setting required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = env_factory()
    N = config.num_envs
    batched = N > 1
    seeds = config.seed+np.arange(N) if batched else config.seed
    obs = _observation(env.reset(seeds), env.observation_dim, N)
    env_state = {k:np.asarray(v).copy() for k,v in env.state().items()}
    project = None
    if method == 'world_model':
        project = env.project_observation
        # Refuse missing/nonstationary state before collecting paid transitions.
        projected = project(jnp.asarray(obs), jnp.zeros((N, env.action_dim) if batched else env.action_dim), jnp.asarray(obs))
        _observation(projected, env.observation_dim, N)
    binding = _binding(env, config, method, precision, env_state)
    carry = _initial(config, env.observation_dim, env.action_dim, env_state)
    carry['obs'] = obs
    elapsed = 0.
    latest = None
    if resume_from is not None:
        _, carry, _, elapsed = checkpoint.restore_checkpoint(Path(resume_from), {}, carry, binding=binding)
        env.restore(carry['env'])
        latest = Path(resume_from)
    ppo, fit, imagined = _kernels(config, env.reward_from_observation, env.action_dim, project)
    policy = jax.jit(lambda pi, norm, obs: _mean(pi, norm, obs))
    began = time.monotonic()
    while int(carry['it']) < config.iterations:
        if stop_requested is not None and stop_requested():
            break
        observations, latents, logps, rewards, dones, values, next_values = [], [], [], [], [], [], []
        # Freeze policy/normalizer for an entire on-policy batch. Every transition
        # is from real env.step, never IK labels or the learned dynamics model.
        for _ in range(config.rollout_steps):
            obs = carry['obs'].copy()
            carry['key'], sample_key = jax.random.split(carry['key'])
            mean = policy(carry['pi'], carry['obs_norm'], jnp.asarray(obs))
            latent = mean+jnp.exp(jnp.clip(carry['pi']['log_std'], -5., 1.))*jax.random.normal(sample_key, mean.shape)
            action = np.asarray(jnp.tanh(latent))
            following, reward, done, _ = env.step(action)
            following = _observation(following, env.observation_dim, N)
            if not np.isfinite(reward).all():
                raise ValueError('environment returned a nonfinite reward')
            observations.append(obs)
            latents.append(np.asarray(latent))
            rewards.append(np.asarray(reward).copy() if batched else float(reward))
            dones.append(np.asarray(done).copy() if batched else bool(done))
            logps.append(np.asarray(_logprob(carry['pi'], mean, latent)) if batched else float(_logprob(carry['pi'], mean, latent)))
            values.append(np.asarray(_value(carry['value'], carry['obs_norm'], obs)) if batched else float(_value(carry['value'], carry['obs_norm'], obs)))
            next_values.append(np.asarray(_value(carry['value'], carry['obs_norm'], following)) if batched else float(_value(carry['value'], carry['obs_norm'], following)))
            for lane in range(N):
                index = int(carry['replay_cursor'])
                for name, value in (('obs', obs), ('action', action), ('next', following), ('reward', reward), ('done', done)):
                    carry['replay_'+name][index] = value[lane] if batched else value
                carry['replay_cursor'] = np.array((index+1)%config.replay_capacity, np.int64)
                carry['replay_size'] = np.array(min(config.replay_capacity, int(carry['replay_size'])+1), np.int64)
            carry['physics_steps'] += N
            if batched:
                indices = np.flatnonzero(done)
                carry['resets'][indices] += 1
                if len(indices):
                    following = _observation(env.reset_at(indices, config.seed+indices+N*carry['resets'][indices]), env.observation_dim, N)
            elif done:
                carry['resets'] += 1
                following = _observation(env.reset(config.seed+int(carry['resets'])), env.observation_dim)
            carry['obs'] = following
        obs_batch = jnp.asarray(np.asarray(observations).reshape(-1, env.observation_dim))
        detail = dict(mean_reward=float(np.mean(rewards)))
        if method == 'ppo':
            advantage, returns = _advantages(rewards, dones, values, next_values, config)
            advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
            for _ in range(config.ppo_epochs):
                carry['pi'], carry['value'], carry['opt_pi'], carry['opt_value'], pl, vl = ppo(
                    carry['pi'], carry['value'], carry['opt_pi'], carry['opt_value'], carry['obs_norm'],
                    obs_batch, jnp.asarray(np.asarray(latents).reshape(-1, env.action_dim)), jnp.asarray(np.asarray(logps).reshape(-1)),
                    jnp.asarray(advantage.reshape(-1)), jnp.asarray(returns.reshape(-1)))
            detail.update(policy_loss=float(pl), value_loss=float(vl))
        else:
            size = int(carry['replay_size'])
            inputs = np.concatenate((carry['replay_obs'][:size], carry['replay_action'][:size]), axis=-1)
            delta = carry['replay_next'][:size]-carry['replay_obs'][:size]
            # Recompute replay statistics without recounting old samples as new.
            carry['wm_buf']['in_norm'] = nets.rn_update(nets.rn_init(inputs.shape[1]), jnp.asarray(inputs))
            carry['wm_buf']['out_norm'] = nets.rn_update(nets.rn_init(delta.shape[1]), jnp.asarray(delta))
            for _ in range(config.wm_updates):
                carry['key'], key = jax.random.split(carry['key'])
                indices = np.asarray(jax.random.randint(key, (config.batch_size,), 0, size))
                carry['wm'], carry['opt_wm'], wl, _ = fit(carry['wm'], carry['opt_wm'], carry['wm_buf'],
                    jnp.asarray(carry['replay_obs'][indices]), jnp.asarray(carry['replay_action'][indices]), jnp.asarray(carry['replay_next'][indices]))
            carry['pi'], carry['opt_pi'], pl, _ = imagined(carry['pi'], carry['opt_pi'], carry['wm'], carry['wm_buf'],
                carry['obs_norm'], jnp.asarray(carry['replay_obs'][indices]))
            detail.update(policy_loss=float(pl), world_model_loss=float(wl))
        # Freeze the observation scale for PPO's lifetime: changing it after its
        # update silently changes the policy without a clipped objective. Env
        # observations already carry explicit physical normalization.
        if not all(np.isfinite(np.asarray(x)).all() for key in ('pi','value','wm','opt_pi','opt_wm','opt_value') for x in jax.tree.leaves(carry[key])):
            raise FloatingPointError('nonfinite learned parameters or optimizer state')
        carry['it'] += 1
        carry['metric_history'][int(carry['it'])-1] = [detail['mean_reward'], detail['policy_loss'],
            detail.get('value_loss', 0.), detail.get('world_model_loss', 0.), int(carry['physics_steps'])]
        carry['env'] = {k:np.asarray(v).copy() for k,v in env.state().items()}
        latest = output/f'checkpoint-{int(carry["it"]):06d}'
        checkpoint.save_checkpoint(latest, {}, carry, binding=binding, iteration=int(carry['it']),
                                   elapsed_seconds=elapsed+time.monotonic()-began)
    if latest is None:
        latest = output/'checkpoint-000000'
        checkpoint.save_checkpoint(latest, {}, carry, binding=binding, iteration=0, elapsed_seconds=elapsed+time.monotonic()-began)
    metrics = [dict(iteration=index+1, mean_reward=float(row[0]), policy_loss=float(row[1]),
                    value_loss=float(row[2]), world_model_loss=float(row[3]), physics_steps=int(row[4]))
               for index, row in enumerate(carry['metric_history'][:int(carry['it'])])]
    receipt = dict(method=('PolicyIK-derived matched PPO' if method=='ppo' else 'XMimic-derived matched world model'),
                   dynamics_model='physical-q-qd/canonical-observation-v1' if method=='world_model' else None,
                   precision=precision, num_envs=N, iterations=int(carry['it']), physics_steps=int(carry['physics_steps']),
                   completed=int(carry['it'])==config.iterations, checkpoint=str(latest), metrics=metrics,
                   elapsed_seconds=elapsed+time.monotonic()-began,
                   scope='CPU-capable matched adaptation, not vanilla reproduction or physical qualification',
                   differences=['common stationary URR environment', 'small MLPs', 'Adam optimizer', 'tanh-bounded Gaussian exploration',
                                'finite-horizon learned-model policy gradient without critic bootstrap' if method=='world_model' else 'full-rollout clipped PPO/GAE batches'])
    return TrainingResult(latest, receipt)


def train(env_factory: Callable[[], Environment], *, method: Literal['ppo', 'world_model'], precision: bool,
          config: LearningConfig, output: Path, resume_from: Path | None = None,
          stop_requested: Callable[[], bool] | None = None) -> TrainingResult:
    """Collect real scalar or batched transitions, always disposing worker pools."""
    env = env_factory()
    if env.binding.get('task', {}).get('partition') == 'evaluation':
        raise ValueError('Evaluation task partition cannot collect learning transitions')
    if config.num_envs > 1:
        if not hasattr(env, 'vectorize'):
            raise ValueError('This environment does not implement batched real physics')
        env = env.vectorize(config.num_envs)
    try:
        return _train(lambda: env, method=method, precision=precision, config=config, output=output,
                      resume_from=resume_from, stop_requested=stop_requested)
    finally:
        if config.num_envs > 1:
            env.close()


def load_policy(path: Path) -> Callable[[np.ndarray], np.ndarray]:
    """Verify the complete retained numeric checkpoint before exposing inference."""
    path = Path(path)
    binding = json.loads((path/'manifest.json').read_text())['binding']
    if binding.get('schema') != 'xmimic.matched-urr-learning/v1':
        raise ValueError('not a matched URR learner checkpoint')
    if binding['code'] != _program() or binding['runtime']['matrix_output'] != str(nets.MM_OUT):
        raise ValueError('checkpoint inference program differs')
    config = LearningConfig(**binding['config'])
    state = {k:np.zeros(v['shape'], dtype=v['dtype']) for k,v in binding['environment_state'].items()}
    template = _initial(config, binding['observation_dim'], binding['action_dim'], state)
    _, carry, _, _ = checkpoint.restore_checkpoint(path, {}, template, binding=binding)
    apply = jax.jit(lambda obs: jnp.tanh(_mean(carry['pi'], carry['obs_norm'], obs)))
    def policy(obs):
        value = _observation(obs, binding['observation_dim'])
        return np.asarray(apply(jnp.asarray(value)))
    return policy
