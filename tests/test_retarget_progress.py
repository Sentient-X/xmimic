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
