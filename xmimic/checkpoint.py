"""Complete synchronized training snapshots, distinct from inference policy NPZs.

No Python objects or tree definitions are deserialized. A fresh trainer supplies the
trusted pytree template (including MJX static fields); numeric leaves are restored only
after exact configuration/input/code binding and structural checks. All dynamic consts
are included, notably per-world randomized model parameters and the reference library.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import tempfile

import jax
import numpy as np


SCHEMA = 'xmimic.training-state/v1'


@dataclass(frozen=True)
class RecoveryOptions:
    directory: str
    resume_from: str | None = None
    stop_file: str | None = None
    # Extra immutable source-selection/retarget receipt identity from the managed caller.
    source_binding: str = ''
    on_checkpoint: Callable[[Path, int], None] | None = None
    on_restored: Callable[[], None] | None = None
    on_metrics: Callable[[dict[str, float | int], int], None] | None = None
    stop_requested: Callable[[], bool] | None = None
    handle_signals: bool = True


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run_binding(cfg, source_binding: str = '') -> dict:
    """Content-bind source motion bytes, complete robot bundle, code and runtime versions.

    Paths and deadlines are operational: content identities replace robot/motion paths;
    log_dir and max_minutes do not change the numerical program. Other Config fields,
    including total iterations and chunk size, must match exactly on resumption.
    """
    config = asdict(cfg)
    for name in ('robot_dir', 'motion_dir', 'log_dir', 'max_minutes'):
        del config[name]
    robot = Path(cfg.robot_dir)
    motion = Path(cfg.motion_dir)
    extension = '*.npz' if cfg.motion_format == 'retargeted_npz' else '*.csv'
    motion_files = sorted(motion.glob(extension))
    if not motion_files:
        raise ValueError('cannot bind an empty motion selection')
    packages = {}
    for package in ('jax', 'jaxlib', 'numpy', 'mujoco', 'mujoco-mjx', 'mujoco-warp', 'warp-lang'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        'configuration': config, 'source_binding': source_binding,
        'embodiment_id': json.loads((robot / 'meta.json').read_text()).get('embodiment_id'),
        'robot_files': {str(p.relative_to(robot)): _digest(p) for p in sorted(robot.rglob('*')) if p.is_file()},
        'motion_files': {p.name: _digest(p) for p in motion_files},
        'program_files': {p.name: _digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'versions': packages,
        'jax_enable_x64': bool(jax.config.jax_enable_x64),
        'backend': jax.default_backend(),
        'devices': [{'platform': d.platform, 'kind': d.device_kind} for d in jax.devices()],
    }


def _leaf_array(leaf):
    key = isinstance(leaf, jax.Array) and jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)
    array = np.asarray(jax.device_get(jax.random.key_data(leaf) if key else leaf))
    if array.dtype.hasobject or array.dtype.kind not in 'biufcV':
        raise ValueError('checkpoint leaves must be numeric arrays')
    return np.ascontiguousarray(array).reshape(array.shape), key


def _static(value):
    """Fingerprint trusted tree metadata without serializing executable objects."""
    kind = type(value).__module__ + '.' + type(value).__qualname__
    if value is None or isinstance(value, (str, int, float, bool)):
        return [kind, value]
    if isinstance(value, bytes):
        return [kind, value.hex()]
    if isinstance(value, np.generic):
        return [kind, _static(value.item())]
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('object arrays cannot be static checkpoint metadata')
        return ['array', str(value.dtype), list(value.shape), hashlib.sha256(value.tobytes()).hexdigest()]
    if kind == 'mujoco.mjx._src.dataclasses._NumPyArrayHashWrapper':
        return [kind, _static(value.array)]
    if is_dataclass(value) and not isinstance(value, type):
        return [kind, [[f.name, _static(getattr(value, f.name))] for f in fields(value)]]
    if isinstance(value, (tuple, list)):
        return [kind, [_static(v) for v in value]]
    if isinstance(value, dict):
        return [kind, [[_static(k), _static(v)] for k, v in sorted(value.items())]]
    if isinstance(value, type):
        return ['type', value.__module__ + '.' + value.__qualname__]
    raise ValueError(f'unsupported static checkpoint metadata type: {kind}')


def _structure(structure):
    node = structure.node_data()
    return {'node': _static(node), 'children': [_structure(child) for child in structure.children()]}


def _spec(tree):
    pairs, structure = jax.tree_util.tree_flatten_with_path(tree)
    leaves, specs = [], []
    for path, leaf in pairs:
        key = isinstance(leaf, jax.Array) and jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)
        array = jax.random.key_data(leaf) if key else leaf
        if not isinstance(array, (jax.Array, np.ndarray)):
            array = np.asarray(array)
        if np.dtype(array.dtype).hasobject or np.dtype(array.dtype).kind not in 'biufcV':
            raise ValueError('checkpoint leaves must be numeric arrays')
        leaves.append(leaf)
        specs.append({'path': jax.tree_util.keystr(path), 'shape': list(array.shape),
                      'dtype': str(array.dtype), 'bytes': array.nbytes, 'key': key,
                      'key_impl': str(jax.random.key_impl(leaf)) if key else None})
    return leaves, specs, structure


def save_checkpoint(path: Path, consts, carry, *, binding: dict, iteration: int,
                    elapsed_seconds: float) -> dict:
    """Block, copy every leaf, fsync, then atomically admit a new immutable directory.

    Failure preserves the previous checkpoint. A partial staging directory is retained
    for diagnosis, never mistaken for a completed checkpoint. Caller must not dispatch
    or donate these buffers concurrently with this function.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if type(iteration) is not int or iteration < 0 or not np.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError('invalid checkpoint boundary')
    if int(np.asarray(jax.device_get(carry['it']))) != iteration:
        raise ValueError('host iteration differs from carry iteration')
    state = {'consts': consts, 'carry': carry}
    jax.block_until_ready(state)
    leaves, specs, structure = _spec(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{path.name}.partial-', dir=path.parent))
    files = []
    for i, (leaf, spec) in enumerate(zip(leaves, specs, strict=True)):
        array, _ = _leaf_array(leaf)
        filename = f'{i:06d}.npy'
        target = staging / filename
        # Byte storage also preserves bfloat16 without NumPy's void-dtype coercion.
        with target.open('xb') as stream:
            np.save(stream, array.reshape(-1).view(np.uint8), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        files.append({**spec, 'file': filename, 'sha256': _digest(target), 'size': target.stat().st_size})
    manifest = {'schema': SCHEMA, 'binding': binding,
                'structure': _structure(structure), 'iteration': iteration,
                'elapsed_seconds': float(elapsed_seconds), 'leaves': files}
    with (staging / 'manifest.json').open('x') as stream:
        stream.write(_json(manifest))
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(staging, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    os.rename(staging, path)
    parent_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return manifest


def restore_checkpoint(path: Path, consts_template, carry_template, *, binding: dict):
    """Restore only into a freshly initialized compatible trainer, without running warmup."""
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('schema') != SCHEMA or _json(manifest.get('binding')) != _json(binding):
        raise ValueError('checkpoint configuration, body, source or program binding differs')
    leaves, specs, structure = _spec({'consts': consts_template, 'carry': carry_template})
    if manifest['structure'] != _structure(structure) or len(manifest['leaves']) != len(specs):
        raise ValueError('checkpoint tree structure differs')
    restored = []
    for i, (template, expected, record) in enumerate(zip(leaves, specs, manifest['leaves'], strict=True)):
        if any(record.get(key) != value for key, value in expected.items()) or record['file'] != f'{i:06d}.npy':
            raise ValueError('checkpoint leaf layout differs')
        source = path / record['file']
        if source.is_symlink() or source.stat().st_size != record['size'] or record['size'] > expected['bytes'] + 4096:
            raise ValueError('checkpoint leaf size differs')
        if _digest(source) != record['sha256']:
            raise ValueError('checkpoint leaf checksum differs')
        raw = np.load(source, allow_pickle=False, mmap_mode='r')
        if raw.dtype != np.uint8 or raw.shape != (expected['bytes'],):
            raise ValueError('checkpoint storage must be a flat byte array')
        template_array, key = _leaf_array(template)
        value = np.array(raw).view(template_array.dtype).reshape(template_array.shape)
        if key:
            value = jax.random.wrap_key_data(value, impl=jax.random.key_impl(template))
        elif isinstance(template, jax.Array):
            value = jax.device_put(value, template.sharding)
        elif not isinstance(template, np.ndarray):
            value = type(template)(value.item())
        restored.append(value)
    state = jax.tree.unflatten(structure, restored)
    iteration, elapsed = manifest['iteration'], manifest['elapsed_seconds']
    if type(iteration) is not int or iteration < 0 or not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError('invalid checkpoint boundary')
    if int(np.asarray(jax.device_get(state['carry']['it']))) != iteration:
        raise ValueError('checkpoint iteration disagrees with restored state')
    return state['consts'], state['carry'], iteration, float(elapsed)
