"""CPU recovery oracles: real JAX optimizer/PRNG/replay and MJX trees, no GPU claim."""
import json
from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import pytest

from xmimic import optim
from xmimic.checkpoint import restore_checkpoint, save_checkpoint
from xmimic.runtime import run_synchronized


def initial(seed):
    k1, k2, key = jax.random.split(jax.random.PRNGKey(seed), 3)
    params = {'weight': jax.random.normal(k1, (3,)), 'bias': jnp.zeros(())}
    mask = {'weight': False, 'bias': False}
    consts = {'mass': jax.random.uniform(k2, (3,), minval=.5, maxval=2.),
              'target': jnp.asarray([.2, -.1, .4]), 'bf16': jnp.asarray([1.1, -2.2], jnp.bfloat16)}
    carry = {'it': jnp.zeros((), jnp.int32), 'key': key, 'params': params, 'opt': optim.init(params, mask),
             'avg': jax.tree.map(jnp.copy, params), 'env': jnp.zeros(3),
             'replay': jnp.zeros((5, 3)), 'normalizer_count': jnp.zeros((), jnp.int32)}
    return consts, carry


@partial(jax.jit, donate_argnums=(1,))
def advance(consts, state):
    key, draw = jax.random.split(state['key'])
    env = state['env'] + .01 * jax.random.normal(draw, (3,)) / consts['mass']
    replay = state['replay'].at[state['it'] % 5].set(env)
    def loss(params):
        prediction = replay @ params['weight'] + params['bias']
        return jnp.mean((prediction - consts['target'].sum()) ** 2)
    params, opt = optim.step(state['params'], jax.grad(loss)(state['params']), state['opt'],
                             {'weight': False, 'bias': False}, .005)
    avg = optim.ema(state['avg'], params, .9, state['it'] == 0)
    return {**state, 'it': state['it'] + 1, 'key': key, 'params': params, 'opt': opt,
            'avg': avg, 'env': env, 'replay': replay, 'normalizer_count': state['normalizer_count'] + 3}


def equal_trees(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


def test_independent_uninterrupted_and_restored_continuation(tmp_path):
    baseline_consts, baseline = initial(32)
    for _ in range(13):
        baseline = advance(baseline_consts, baseline)
    consts, split = initial(32)
    for _ in range(6):
        split = advance(consts, split)
    path = tmp_path / 'state'
    save_checkpoint(path, consts, split, binding={'body': 'test', 'source': 'abc'}, iteration=6, elapsed_seconds=2.)
    # Different initialization demonstrates randomized constants and optimizer/RNG state
    # are restored, not accidentally regenerated from the original initialization.
    fresh_consts, fresh = initial(99)
    restored_consts, restored, it, elapsed = restore_checkpoint(path, fresh_consts, fresh,
                                                               binding={'body': 'test', 'source': 'abc'})
    assert (it, elapsed) == (6, 2.)
    equal_trees(consts, restored_consts)
    for _ in range(it, 13):
        restored = advance(restored_consts, restored)
    equal_trees(baseline, restored)


def test_real_mjx_dataclasses_and_typed_keys_roundtrip(tmp_path):
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>')
    consts = {'model': mjx.put_model(model)}
    carry = {'it': jnp.asarray(0, jnp.int32), 'data': mjx.make_data(model), 'key': jax.random.key(123)}
    save_checkpoint(tmp_path / 'mjx', consts, carry, binding={}, iteration=0, elapsed_seconds=0.)
    c, s, _, _ = restore_checkpoint(tmp_path / 'mjx', {'model': mjx.put_model(model)},
                                    {'it': jnp.asarray(0, jnp.int32), 'data': mjx.make_data(model),
                                     'key': jax.random.key(999)}, binding={})
    np.testing.assert_array_equal(jax.random.normal(s['key'], (4,)), jax.random.normal(carry['key'], (4,)))
    equal_trees(c['model'], consts['model'])
    equal_trees(s['data'], carry['data'])


@pytest.mark.parametrize('mutation', ['binding', 'checksum', 'layout', 'missing', 'iteration', 'path'])
def test_incompatible_or_corrupt_checkpoint_refused(tmp_path, mutation):
    consts, carry = initial(1)
    path = tmp_path / 'state'
    save_checkpoint(path, consts, carry, binding={'source': 'a'}, iteration=0, elapsed_seconds=0.)
    binding = {'source': 'a'}
    manifest = json.loads((path / 'manifest.json').read_text())
    first = path / manifest['leaves'][0]['file']
    if mutation == 'binding':
        binding['source'] = 'b'
    elif mutation == 'checksum':
        data = bytearray(first.read_bytes())
        data[-1] ^= 1
        first.write_bytes(data)
    elif mutation == 'layout':
        manifest['leaves'][0]['shape'] = [123]
    elif mutation == 'missing':
        first.unlink()  # disposable test file, never raw data
    elif mutation == 'iteration':
        manifest['iteration'] = 1
    else:
        manifest['leaves'][0]['file'] = '../other.npy'
    (path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        restore_checkpoint(path, consts, carry, binding=binding)


def test_admitted_snapshot_never_overwritten(tmp_path):
    consts, carry = initial(1)
    path = tmp_path / 'state'
    save_checkpoint(path, consts, carry, binding={}, iteration=0, elapsed_seconds=0.)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, consts, carry, binding={}, iteration=0, elapsed_seconds=1.)


def test_synchronized_loop_stops_before_next_dispatch_and_saves_final_boundary():
    events = []
    class Log:
        def submitted(self, it, stats):
            events.append(('submitted', it))
        def add(self, it, stats, avg):
            events.append(('logged', it))
            return float(it)
        def close(self):
            events.append(('closed',))
    count = [4]
    def dispatch():
        count[0] += 2
        events.append(('dispatch', count[0]))
        return count[0], {'a': jnp.ones(2)}, jnp.zeros(2)
    it = run_synchronized(dispatch, Log(), 20, start_iteration=4, max_minutes=7,
                          checkpoint_every=6, checkpoint=lambda it: events.append(('saved', it)))
    assert it == 8
    assert events == [('dispatch', 6), ('submitted', 6), ('logged', 6), ('saved', 6),
                      ('dispatch', 8), ('submitted', 8), ('logged', 8), ('saved', 8), ('closed',)]


def test_binding_tracks_motion_bytes_robot_bytes_and_numerical_config(tmp_path):
    from xmimic.checkpoint import run_binding
    from xmimic.config import Config
    robot, motions = tmp_path / 'robot', tmp_path / 'motions'
    robot.mkdir()
    motions.mkdir()
    (robot / 'meta.json').write_text('{"embodiment_id":"fixture"}')
    (robot / 'scene.xml').write_text('fixture scene bytes')
    (motions / 'clip.npz').write_bytes(b'fixture motion bytes')
    cfg = Config(robot_dir=str(robot), motion_dir=str(motions), motion_format='retargeted_npz')
    original = run_binding(cfg, 'selected-source-receipt')
    cfg.log_dir, cfg.max_minutes = 'new-runtime-location', 2.
    assert original == run_binding(cfg, 'selected-source-receipt')
    cfg.seed = 4
    assert original != run_binding(cfg, 'selected-source-receipt')
    cfg.seed = 0
    (motions / 'clip.npz').write_bytes(b'other motion bytes')
    assert original['motion_files'] != run_binding(cfg, 'selected-source-receipt')['motion_files']
    (robot / 'scene.xml').write_text('other body geometry')
    assert original['robot_files'] != run_binding(cfg, 'selected-source-receipt')['robot_files']
