"""Durable frame boundaries for one exact CPU retarget execution.

A frame becomes reusable only after its complete coordinates, diagnostics and hash
chain are fsynced. A kill during the current frame may repeat that uncommitted frame;
no committed frame is solved again. This journal stays on the managed run volume,
while the final motion/diagnostics use ordinary Catalog publication.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .artifact_cache import ArtifactCache, CacheIdentity, CacheError, _atomic, _json, _read, _require, content_hash
from .retarget_target import Disposition, RetargetFrame


SCHEMA = 'xmimic.retarget-progress/v2'
LEGACY_SCHEMA = 'xmimic.retarget-progress/v1'


class FrameJournal:
    """Use only under the cache entry's process lock; never adopt another execution."""

    def __init__(self, path: Path, *, binding: dict, total_frames: int,
                 joint_count: int, landmark_count: int, create: bool, direction_count: int = 0,
                 assert_active: Callable[[], None]):
        self.path, self._assert_active = path, assert_active
        self._assert_active()
        _require(all(type(n) is int and n > 0 for n in (total_frames, joint_count))
                 and all(type(n) is int and n >= 0 for n in (landmark_count, direction_count))
                 and landmark_count + direction_count > 0,
                 'Progress dimensions require positive frames/joints and nonnegative nonempty targets')
        _require(not path.is_symlink(), 'Linked progress directory')
        header = {'schema': SCHEMA, 'binding': binding, 'total_frames': total_frames,
                  'joint_count': joint_count, 'landmark_count': landmark_count, 'direction_count': direction_count}
        header_path = path/'binding.json'
        if not header_path.exists():
            _require(create, 'Running clip has no retained frame journal; reconcile its execution')
            # A killed first header write contains no accepted solver work.
            path.mkdir(exist_ok=True)
            _require(not list(path.glob('frame-*.json')), 'Progress frames have no binding')
            _atomic(header_path, _json(header))
        retained = _read(header_path)
        if retained.get('schema') == LEGACY_SCHEMA and direction_count == 0:
            # Keep the original immutable header and hash chain. A v1 journal
            # can resume zero-direction work, never satisfy a new direction target.
            header = {key: value for key, value in header.items() if key != 'direction_count'}
            header['schema'] = LEGACY_SCHEMA
        _require(retained == header, 'Retarget progress belongs to another execution or shape')
        self.total_frames, self.joint_count, self.landmark_count = total_frames, joint_count, landmark_count
        self.direction_count = direction_count
        self._checkpoint_schema = ('xmimic.retarget-checkpoint/v1' if header['schema'] == LEGACY_SCHEMA
                                   else 'xmimic.retarget-checkpoint/v2')
        self._header = header
        self._previous = content_hash(header)
        self._digests = [self._previous]
        self._frames: list[RetargetFrame] = []
        tip_path = path/'tip.json'
        if not tip_path.exists():
            _require(create and not list(path.glob('frame-*.json')),
                     'Retarget progress lost its durable tip; refuse rollback')
            self._write_tip()
        tip_envelope = _read(tip_path)
        _require(set(tip_envelope) == {'sha256', 'tip'}, 'Malformed progress tip')
        tip = tip_envelope['tip']
        _require(isinstance(tip, dict) and set(tip) == {'binding_sha256', 'completed', 'prefix_sha256'}
                 and content_hash(tip) == tip_envelope['sha256']
                 and tip['binding_sha256'] == self._digests[0]
                 and type(tip['completed']) is int and 0 <= tip['completed'] <= total_frames,
                 'Retarget progress tip is corrupt or belongs to another binding')
        for index, item in enumerate(sorted(path.glob('frame-*.json'))):
            _require(item.name == self._filename(index), 'Retarget progress has a missing or invalid frame')
            envelope = _read(item)
            _require(set(envelope) == {'sha256', 'frame'}, 'Malformed frame envelope')
            record = envelope['frame']
            _require(isinstance(record, dict) and content_hash(record) == envelope['sha256'],
                     'Retarget frame digest differs')
            self._frames.append(self._parse(record, index))
            self._previous = envelope['sha256']
            self._digests.append(self._previous)
        _require(len(self._frames) <= total_frames, 'Retained progress exceeds sequence')
        _require(tip['completed'] <= len(self._frames),
                 'Retarget progress lost a committed tail frame; refuse rollback')
        _require(self._digests[tip['completed']] == tip['prefix_sha256'], 'Retarget tip prefix digest differs')
        if tip['completed'] < len(self._frames):
            # A crash after the immutable frame but before its tip retained valid
            # work. Finish that commit before acknowledging the recovered prefix.
            self._write_tip()

    def _write_tip(self) -> None:
        self._assert_active()
        tip = {'binding_sha256': self._digests[0], 'completed': len(self._frames),
               'prefix_sha256': self._previous}
        _atomic(self.path/'tip.json', _json({'sha256': content_hash(tip), 'tip': tip}))

    @staticmethod
    def _filename(index: int) -> str:
        return f'frame-{index:09d}.json'

    @property
    def frames(self) -> tuple[RetargetFrame, ...]:
        self._assert_active()
        return tuple(self._frames)

    def _parse(self, record: dict, index: int) -> RetargetFrame:
        fields = {'index', 'previous_sha256', 'q', 'position_residual_m',
                  'orientation_residual_rad', 'iterations', 'disposition'}
        if self._header['schema'] == SCHEMA:
            fields.add('direction_residual_rad')
        _require(set(record) == fields, 'Malformed retained frame')
        _require(type(record['index']) is int and record['index'] == index
                 and record['previous_sha256'] == self._previous,
                 'Retarget frame order or hash chain differs')
        arrays = []
        for name, count in [('q', self.joint_count), ('position_residual_m', self.landmark_count),
                            ('orientation_residual_rad', self.landmark_count),
                            ('direction_residual_rad', self.direction_count)]:
            values = record.get(name, []) if name == 'direction_residual_rad' else record[name]
            _require(isinstance(values, list) and len(values) == count
                     and all(type(v) in (int, float) for v in values), 'Malformed retained numeric frame')
            value = np.array(values, dtype=np.float64)
            _require(np.isfinite(value).all() and (name == 'q' or np.all(value >= 0)),
                     'Invalid retained numeric frame')
            value.flags.writeable = False
            arrays.append(value)
        _require(type(record['iterations']) is int and record['iterations'] >= 0,
                 'Invalid retained iteration count')
        try:
            disposition = Disposition(record['disposition'])
        except (ValueError, TypeError) as error:
            raise CacheError('Invalid retained frame disposition') from error
        return RetargetFrame(index, *arrays[:3], record['iterations'], disposition,
                             direction_residual_rad=arrays[3])

    def append(self, frame: RetargetFrame) -> None:
        self._assert_active()
        _require(frame.index == len(self._frames) and frame.index < self.total_frames,
                 'Progress must append exactly the next frame')
        record = {'index': frame.index, 'previous_sha256': self._previous,
                  'q': frame.q.tolist(), 'position_residual_m': frame.position_residual_m.tolist(),
                  'orientation_residual_rad': frame.orientation_residual_rad.tolist(),
                  'iterations': frame.iterations, 'disposition': frame.disposition.value}
        if self._header['schema'] == SCHEMA:
            record['direction_residual_rad'] = frame.direction_residual_rad.tolist()
        else:
            _require(frame.direction_residual_rad.shape == (0,), 'Legacy journal cannot retain direction evidence')
        parsed = self._parse(record, frame.index)
        digest = content_hash(record)
        payload = _json({'sha256': digest, 'frame': record})
        destination = self.path/self._filename(frame.index)
        _require(not destination.exists() and not destination.is_symlink(), 'Committed frame cannot be replaced')
        _atomic(destination, payload)
        self._frames.append(parsed)
        self._previous = digest
        self._digests.append(digest)
        self._write_tip()  # No callback acknowledges a frame before its durable count/hash.

    def checkpoint_receipt(self) -> dict:
        """Opaque managed checkpoint binds an acknowledged prefix on the run volume."""
        self._assert_active()
        return {**self._header, 'schema': self._checkpoint_schema,
                'completed': len(self._frames), 'prefix_sha256': self._previous}

    def verify_checkpoint(self, receipt: dict) -> None:
        self._assert_active()
        _require(set(receipt) == {*self._header, 'completed', 'prefix_sha256'}
                 and receipt['schema'] == self._checkpoint_schema, 'Invalid CPU checkpoint receipt')
        expected = {**self._header, 'schema': self._checkpoint_schema}
        _require(all(receipt[key] == value for key, value in expected.items()), 'CPU checkpoint binding differs')
        count = receipt['completed']
        _require(type(count) is int and 0 <= count <= len(self._frames),
                 'CPU checkpoint acknowledged progress is missing; refuse rollback')
        _require(self._digests[count] == receipt['prefix_sha256'], 'CPU checkpoint prefix digest differs')


def verify_progress_checkpoint(cache: ArtifactCache, receipt: dict, *, clip_id: str, execution_ref: str) -> None:
    """Verify the portable envelope against retained native state under the same cache lock.

    The native journal may be ahead of the last managed receipt. It may never be
    behind it. A receipt is not a replacement for missing paid solver coordinates.
    """
    _require(isinstance(receipt, dict), 'Invalid CPU checkpoint receipt')
    fields = {'schema', 'binding', 'total_frames', 'joint_count', 'landmark_count', 'completed', 'prefix_sha256'}
    schema = receipt.get('schema')
    _require(schema in ('xmimic.retarget-checkpoint/v1', 'xmimic.retarget-checkpoint/v2'),
             'Invalid CPU checkpoint schema')
    if schema == 'xmimic.retarget-checkpoint/v2':
        fields.add('direction_count')
    _require(set(receipt) == fields, 'Invalid CPU checkpoint receipt')
    binding = receipt['binding']
    _require(isinstance(binding, dict) and set(binding) == {'cache_key', 'clip_id', 'execution_ref'}
             and binding['clip_id'] == clip_id and binding['execution_ref'] == execution_ref,
             'CPU checkpoint execution differs')
    from .artifact_cache import _sha
    key = _sha(binding['cache_key'])
    directory = cache.root/key
    _require(not directory.is_symlink(), 'Linked native cache')
    state = _read(directory/'state.json')
    try:
        identity = CacheIdentity(**state['document']['identity'])
    except (KeyError, TypeError) as error:
        raise CacheError('Missing native cache identity') from error
    _require(identity.key == key, 'CPU checkpoint cache identity differs')
    with cache.entry(identity, [clip_id]) as entry:
        clip = entry.document['clips'][clip_id]
        _require(clip['execution_ref'] == execution_ref and clip['status'] != 'pending',
                 'CPU checkpoint execution has no retained work')
        journal = FrameJournal(directory/('progress-' + content_hash({'clip_id': clip_id})),
                               binding=binding, total_frames=receipt['total_frames'],
                               joint_count=receipt['joint_count'], landmark_count=receipt['landmark_count'],
                               direction_count=receipt.get('direction_count', 0),
                               create=False, assert_active=lambda: _require(entry._active, 'Cache lock released'))
        journal.verify_checkpoint(receipt)
