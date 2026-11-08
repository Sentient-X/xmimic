"""Real process death preserves exactly the completed CPU solver prefix."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from test_retarget_target import fixture_body
from test_retarget_workflow import recipe, source
from xmimic.artifact_cache import ArtifactCache, CacheError
from xmimic.retarget_target import PreparedTarget, RetargetInterrupted
from xmimic.retarget_workflow import retarget_clip


def run(directory, monkeypatch, *, rejected=False, **options):
    directory.mkdir(exist_ok=True)
    body = fixture_body(directory, monkeypatch)
    return retarget_clip(body, source(rejected)[0], recipe(),
                         expected_embodiment_id=str(body.id), cache=ArtifactCache(directory/'cache'),
                         clip_id='clip', source_group='family', execution_ref='retained-cpu-job', **options)


def artifacts(receipt):
    manifest = json.loads((receipt.directory/'manifest.json').read_bytes())
    return {name: receipt.directory/'blobs'/ref['sha256']
            for name, ref in manifest['clips']['clip']['artifacts'].items()}


@pytest.mark.parametrize('rejected', [False, True])
def test_process_kill_resumes_exact_prefix_without_solving_committed_frames(tmp_path, monkeypatch, rejected):
    reference = run(tmp_path/'reference', monkeypatch, rejected=rejected)
    resumed_dir = tmp_path/'interrupted'
    program = '''
import os,sys
from pathlib import Path
from pytest import MonkeyPatch
from test_retarget_progress import run
with MonkeyPatch.context() as patch:
    run(Path(sys.argv[1]),patch,rejected=sys.argv[2]=='yes',
        on_progress=lambda completed,total: os._exit(23) if completed==3 else None)
'''
    environment = {**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)}
    stopped = subprocess.run([sys.executable, '-c', program, str(resumed_dir), 'yes' if rejected else 'no'],
                             env=environment, capture_output=True, text=True, timeout=30)
    assert stopped.returncode == 23, stopped.stderr
    journal, = list((resumed_dir/'cache').glob('*/progress-*'))
    assert len(list(journal.glob('frame-*.json'))) == 3
    original = PreparedTarget.poses
    seen = []
    def forbid_prior_frames(self, q, root_p, root_quat):
        # Source x is 0.10, 0.11, 0.12 for the three already paid frames.
        assert root_p[0] >= .13 - 1e-12, 'solver repeated an acknowledged frame'
        seen.append(float(root_p[0]))
        return original(self, q, root_p, root_quat)
    with monkeypatch.context() as patch:
        patch.setattr(PreparedTarget, 'poses', forbid_prior_frames)
        completed = []
        result = run(resumed_dir, patch, rejected=rejected,
                     on_progress=lambda count,total: completed.append(count))
    assert completed == [4,5,6,7,8] and seen
    assert result.disposition == reference.disposition
    expected, actual = artifacts(reference), artifacts(result)
    assert expected.keys() == actual.keys()
    assert json.loads(actual['retarget.json'].read_bytes()) == json.loads(expected['retarget.json'].read_bytes())
    name = 'candidate.npz' if rejected else 'motion.npz'
    with np.load(expected[name], allow_pickle=False) as before, np.load(actual[name], allow_pickle=False) as after:
        assert before.files == after.files
        for field in before.files:
            np.testing.assert_array_equal(before[field], after[field])
    assert len(list(journal.glob('frame-*.json'))) == 8


def stop_after_two(directory, monkeypatch):
    committed = []
    with pytest.raises(RetargetInterrupted) as error:
        run(directory, monkeypatch, stop_requested=lambda: len(committed) == 2,
            on_progress=lambda count,total: committed.append(count))
    assert error.value.completed == 2 and error.value.total == 8
    return next((directory/'cache').glob('*/progress-*'))


@pytest.mark.parametrize('mutation', ['bytes', 'gap', 'binding', 'shape'])
def test_changed_or_corrupt_progress_refuses_without_solver(tmp_path, monkeypatch, mutation):
    journal = stop_after_two(tmp_path/'run', monkeypatch)
    if mutation == 'bytes':
        frame = journal/'frame-000000000.json'
        payload = json.loads(frame.read_bytes())
        payload['frame']['q'][0] += .01
        frame.write_text(json.dumps(payload))
    elif mutation == 'gap':
        (journal/'frame-000000000.json').rename(journal/'displaced-for-test.json')
    else:
        path = journal/'binding.json'
        header = json.loads(path.read_bytes())
        if mutation == 'binding':
            header['binding']['execution_ref'] = 'another-job'
        else:
            header['total_frames'] += 1
        path.write_text(json.dumps(header))
    def forbidden(*args, **kwargs):
        pytest.fail('corrupt progress must be refused before solver')
    monkeypatch.setattr('xmimic.retarget_workflow.solve_sequence', forbidden)
    with pytest.raises(CacheError):
        run(tmp_path/'run', monkeypatch)


def test_cooperative_stop_before_first_frame_can_resume_empty_prefix(tmp_path, monkeypatch):
    with pytest.raises(RetargetInterrupted, match='0/8'):
        run(tmp_path/'run', monkeypatch, stop_requested=lambda: True)
    result = run(tmp_path/'run', monkeypatch)
    assert result.disposition.value == 'succeeded'


def test_settled_clip_finishes_manifest_without_recomputing(tmp_path, monkeypatch):
    from xmimic.artifact_cache import CacheEntry
    with monkeypatch.context() as patch:
        patch.setattr(CacheEntry, 'complete', lambda self: (_ for _ in ()).throw(InterruptedError()))
        with pytest.raises(InterruptedError):
            run(tmp_path/'run', patch)
    monkeypatch.setattr('xmimic.retarget_workflow.solve_sequence',
                        lambda *a, **k: pytest.fail('settled clip was solved again'))
    assert run(tmp_path/'run', monkeypatch).disposition.value == 'succeeded'


def stopped_runtime(tmp_path, monkeypatch):
    from sx_training import TrainingInterruptedError, execute
    from test_managed_worker import context_fixture
    from xmimic.retarget_progress import FrameJournal
    context = context_fixture(tmp_path, monkeypatch)
    stopped = []
    context._cancelled = lambda: bool(stopped)
    append = FrameJournal.append
    def stop_at_three(self, frame):
        append(self, frame)
        if frame.index == 2:
            stopped.append(True)
    with monkeypatch.context() as patch:
        patch.setattr(FrameJournal, 'append', stop_at_three)
        with pytest.raises(TrainingInterruptedError):
            execute(context)
    assert (context.run_dir/'checkpoint_000000000003.pt').is_file()
    return context


def reenter(context):
    from sx_training import TrainingContext, execution_bytes
    return TrainingContext(execution_bytes(context.inputs, context.trainer),
                           data_root=context.data_root, cache_dir=context.cache_dir,
                           runs_dir=context.run_dir.parent)


@pytest.mark.parametrize('native_ahead', [False, True])
def test_actual_managed_runtime_restores_exact_cpu_prefix(tmp_path, monkeypatch, native_ahead):
    from sx_training import execute, verify_receipt
    from sx_contracts.identity import content_digest
    context = stopped_runtime(tmp_path, monkeypatch)
    if native_ahead:
        # Model a kill after the native frame commit and before envelope checkpoint.
        checkpoint = context.run_dir/'checkpoint_000000000003.pt'
        checkpoint.rename(context.run_dir/'not-yet-admitted-envelope')
    fresh = reenter(context)
    original = PreparedTarget.poses
    def forbid_repeat(self, q, root_p, root_quat):
        assert root_p[0] >= .13 - 1e-12
        return original(self, q, root_p, root_quat)
    monkeypatch.setattr(PreparedTarget, 'poses', forbid_repeat)
    output = execute(fresh)
    assert output == fresh.run_dir/'artifact'
    verify_receipt(output, spec_digest=content_digest(fresh.inputs['spec']))
    assert (fresh.run_dir/'checkpoint_000000000008.pt').is_file()


@pytest.mark.parametrize('missing', ['journal', 'acknowledged_frame'])
def test_managed_runtime_refuses_missing_acknowledged_native_state(tmp_path, monkeypatch, missing):
    from sx_training import ResumeRefusedError, execute
    context = stopped_runtime(tmp_path, monkeypatch)
    journal, = list((context.run_dir/'xmimic/retarget-cache').glob('*/progress-*'))
    if missing == 'journal':
        journal.rename(journal.with_name('withheld-native-state'))
    else:
        frame = journal/'frame-000000002.json'
        frame.rename(journal/'withheld-frame')
    monkeypatch.setattr('xmimic.retarget_workflow.solve_sequence',
                        lambda *a, **k: pytest.fail('missing paid progress must never restart'))
    with pytest.raises(ResumeRefusedError):
        execute(reenter(context))


def test_committed_tail_loss_refuses_even_before_first_periodic_platform_checkpoint(tmp_path, monkeypatch):
    from sx_training import ResumeRefusedError, execute, execution_bytes
    from test_managed_worker import context_fixture
    context = context_fixture(tmp_path, monkeypatch)
    document = tmp_path/'invocation.json'
    document.write_bytes(execution_bytes(context.inputs, context.trainer))
    program = '''
import os,sys
from pathlib import Path
from sx_training import TrainingContext, execute
from xmimic.retarget_progress import FrameJournal
context = TrainingContext(Path(sys.argv[1]).read_bytes(), data_root=Path(sys.argv[2]),
                          cache_dir=Path(sys.argv[3]), runs_dir=Path(sys.argv[4]))
append = FrameJournal.append
def kill_after_two(self,frame):
    append(self,frame)
    if frame.index == 1:
        os._exit(43)
FrameJournal.append = kill_after_two
execute(context)
'''
    killed = subprocess.run([sys.executable, '-c', program, str(document), str(context.data_root),
                             str(context.cache_dir), str(context.run_dir.parent)],
                            env={**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)},
                            capture_output=True, text=True, timeout=30)
    assert killed.returncode == 43, killed.stderr
    assert [p.name for p in context.run_dir.glob('checkpoint*.pt')] == ['checkpoint_000000000000.pt']
    journal, = list((context.run_dir/'xmimic/retarget-cache').glob('*/progress-*'))
    assert json.loads((journal/'tip.json').read_bytes())['tip']['completed'] == 2
    (journal/'frame-000000001.json').rename(journal/'missing-tail-for-test')
    monkeypatch.setattr('xmimic.retarget_workflow.solve_sequence',
                        lambda *a, **k: pytest.fail('lost committed tail must never rerun'))
    with pytest.raises(ResumeRefusedError, match='committed tail'):
        execute(reenter(context))


def test_frame_committed_before_tip_failure_is_recovered_without_recompute(tmp_path, monkeypatch):
    from xmimic.retarget_progress import FrameJournal
    original_tip = FrameJournal._write_tip
    def fail_second_tip(self):
        if len(self.frames) == 2:
            raise InterruptedError('frame committed; tip not yet acknowledged')
        original_tip(self)
    with monkeypatch.context() as patch:
        patch.setattr(FrameJournal, '_write_tip', fail_second_tip)
        with pytest.raises(InterruptedError):
            run(tmp_path/'run', patch)
    journal, = list((tmp_path/'run/cache').glob('*/progress-*'))
    assert len(list(journal.glob('frame-*.json'))) == 2
    assert json.loads((journal/'tip.json').read_bytes())['tip']['completed'] == 1
    original_poses = PreparedTarget.poses
    def forbid_repeat(self, q, root_p, root_quat):
        assert root_p[0] >= .12 - 1e-12
        return original_poses(self, q, root_p, root_quat)
    monkeypatch.setattr(PreparedTarget, 'poses', forbid_repeat)
    assert run(tmp_path/'run', monkeypatch).disposition.value == 'succeeded'
    assert json.loads((journal/'tip.json').read_bytes())['tip']['completed'] == 8


def run_directions(directory, monkeypatch, *, kill=False):
    from test_artifact_cache import identity
    from test_retarget_target import inputs
    from xmimic.retarget_target import Direction, IKConfig, TargetProfile, prepare_target, solve_sequence
    directory.mkdir(exist_ok=True)
    body = fixture_body(directory, monkeypatch)
    profile = TargetProfile((), holds=(('q3', .3),), directions=(
        Direction('shoulder', 'elbow', 'a', 'b'), Direction('elbow', 'hand', 'b', 'tip')))
    target = prepare_target(body, profile, expected_embodiment_id=str(body.id))
    args, q1, q2 = inputs()
    yaw = np.linspace(-.3, .2, 8)
    directions = {key: np.column_stack((np.cos(angle), np.sin(angle), np.zeros(8)))
                  for key, angle in [('shoulder->elbow', q1+yaw), ('elbow->hand', q1+q2+yaw)]}
    cache = ArtifactCache(directory/'cache')
    with cache.entry(identity(), ['clip']) as entry:
        journal = entry.frame_progress('clip', 'direction-job', total_frames=8, joint_count=3,
                                       landmark_count=0, direction_count=2)
        def committed(frame):
            journal.append(frame)
            if kill and frame.index == 2:
                os._exit(47)
        result = solve_sequence(target, {}, root_p=args['root_p'], root_quat=args['root_quat'],
                                fps=30, initial_q=args['initial_q'], directions=directions,
                                config=IKConfig(direction_tolerance_rad=1e-6),
                                resume=journal.frames, on_frame=committed)
        receipt = journal.checkpoint_receipt()
    return result, receipt, cache


def test_direction_only_process_kill_retains_angles_and_resumes_without_paid_prefix(tmp_path, monkeypatch):
    from xmimic.retarget_progress import verify_progress_checkpoint
    reference, _, _ = run_directions(tmp_path/'reference', monkeypatch)
    interrupted = tmp_path/'interrupted'
    program = '''
import sys
from pathlib import Path
from pytest import MonkeyPatch
from test_retarget_progress import run_directions
with MonkeyPatch.context() as patch:
    run_directions(Path(sys.argv[1]), patch, kill=True)
'''
    stopped = subprocess.run([sys.executable, '-c', program, str(interrupted)],
                             env={**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)},
                             capture_output=True, text=True, timeout=30)
    assert stopped.returncode == 47, stopped.stderr
    original = PreparedTarget.poses
    def forbid_repeat(self, q, root_p, root_quat):
        assert root_p[0] >= .1 + .2 * 3/7 - 1e-12, 'repeated a committed directional frame'
        return original(self, q, root_p, root_quat)
    with monkeypatch.context() as patch:
        patch.setattr(PreparedTarget, 'poses', forbid_repeat)
        resumed, receipt, cache = run_directions(interrupted, patch)
    np.testing.assert_array_equal(resumed.q, reference.q)
    np.testing.assert_array_equal(resumed.direction_residual_rad, reference.direction_residual_rad)
    assert resumed.direction_residual_rad.shape == (8, 2)
    assert receipt['schema'] == 'xmimic.retarget-checkpoint/v2' and receipt['direction_count'] == 2
    verify_progress_checkpoint(cache, receipt, clip_id='clip', execution_ref='direction-job')
    with pytest.raises(CacheError, match='shape'):
        verify_progress_checkpoint(cache, {**receipt, 'direction_count': 1},
                                   clip_id='clip', execution_ref='direction-job')


def test_v1_native_journal_and_checkpoint_remain_readable_without_rewriting(tmp_path):
    from test_artifact_cache import identity
    from xmimic.artifact_cache import content_hash
    from xmimic.retarget_progress import verify_progress_checkpoint
    cache = ArtifactCache(tmp_path/'cache')
    with cache.entry(identity(), ['clip']) as entry:
        entry.start_clip('clip', 'old-job')
        path = entry.path/('progress-'+content_hash({'clip_id': 'clip'}))
        path.mkdir()
        header = {'schema': 'xmimic.retarget-progress/v1',
                  'binding': {'cache_key': identity().key, 'clip_id': 'clip', 'execution_ref': 'old-job'},
                  'total_frames': 2, 'joint_count': 1, 'landmark_count': 1}
        frame = {'index': 0, 'previous_sha256': content_hash(header), 'q': [.2],
                 'position_residual_m': [.001], 'orientation_residual_rad': [0.],
                 'iterations': 3, 'disposition': 'geometrically_feasible'}
        tip = {'binding_sha256': content_hash(header), 'completed': 1, 'prefix_sha256': content_hash(frame)}
        for name, payload in [('binding.json', header), ('frame-000000000.json', {'sha256': content_hash(frame), 'frame': frame}),
                              ('tip.json', {'sha256': content_hash(tip), 'tip': tip})]:
            (path/name).write_text(json.dumps(payload))
        before = {p.name: p.read_bytes() for p in path.iterdir()}
        journal = entry.frame_progress('clip', 'old-job', total_frames=2, joint_count=1, landmark_count=1)
        assert journal.frames[0].direction_residual_rad.shape == (0,)
        receipt = journal.checkpoint_receipt()
        assert receipt['schema'] == 'xmimic.retarget-checkpoint/v1' and 'direction_count' not in receipt
        with pytest.raises(CacheError, match='shape'):
            entry.frame_progress('clip', 'old-job', total_frames=2, joint_count=1, landmark_count=1, direction_count=1)
    verify_progress_checkpoint(cache, receipt, clip_id='clip', execution_ref='old-job')
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before


@pytest.mark.parametrize('fault', ['missing_angles', 'missing_record_field', 'nonfinite', 'negative_count', 'bool_count'])
def test_direction_journal_refuses_missing_or_invalid_evidence(tmp_path, fault):
    from test_artifact_cache import identity
    from xmimic.artifact_cache import content_hash
    from xmimic.retarget_target import RetargetFrame, Disposition
    cache = ArtifactCache(tmp_path/'cache')
    with cache.entry(identity(), ['clip']) as entry:
        if fault.endswith('count'):
            with pytest.raises(CacheError, match='dimensions'):
                entry.frame_progress('clip', 'job', total_frames=2, joint_count=1, landmark_count=0,
                                     direction_count=-1 if fault == 'negative_count' else True)
            return
        journal = entry.frame_progress('clip', 'job', total_frames=2, joint_count=1, landmark_count=0, direction_count=1)
        frame = RetargetFrame(0, np.array([.2]), np.array([]), np.array([]), 1, Disposition.GEOMETRICALLY_FEASIBLE,
                              direction_residual_rad=np.array([] if fault == 'missing_angles' else [np.nan if fault == 'nonfinite' else .01]))
        if fault != 'missing_record_field':
            with pytest.raises(CacheError, match='numeric'):
                journal.append(frame)
            assert journal.frames == ()
        else:
            journal.append(frame)
            record_path = journal.path/'frame-000000000.json'
            payload = json.loads(record_path.read_bytes())
            del payload['frame']['direction_residual_rad']
            payload['sha256'] = content_hash(payload['frame'])
            record_path.write_text(json.dumps(payload))
            with pytest.raises(CacheError, match='Malformed retained frame'):
                entry.frame_progress('clip', 'job', total_frames=2, joint_count=1, landmark_count=0, direction_count=1)
