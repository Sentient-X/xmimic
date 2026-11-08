import json
import os
from pathlib import Path
import tarfile

import pytest

from tools.publish_urr_benchmark import archive_files, checkpoint_files, digest, prepare, verify_archive
from xmimic.artifact_cache import content_hash


def test_archive_preserves_empty_members_and_refuses_mutated_inventory(tmp_path):
    empty = tmp_path/'empty.py'
    empty.write_bytes(b'')
    source = tmp_path/'source.py'
    source.write_bytes(b'original source\n')
    archive = tmp_path/'evidence.tar.gz'
    files = {'package/empty.py':empty,'package/source.py':source}
    inventory = archive_files(files,archive)
    assert inventory['package/empty.py']['size_bytes'] == 0
    verify_archive(archive,inventory)
    other = tmp_path/'same.tar.gz'
    archive_files(files,other)
    assert archive.read_bytes() == other.read_bytes()
    source.write_bytes(b'changed source\n')
    with pytest.raises(ValueError,match='inventory differs'):
        archive_files(files,archive)
    with pytest.raises(ValueError,match='Unsafe'):
        archive_files({'../escaped':empty},tmp_path/'bad.tar.gz')
    assert not (tmp_path/'bad.tar.gz').exists()


def test_only_final_complete_checkpoint_is_selected_without_touching_older(tmp_path):
    old = tmp_path/'checkpoint-000001'
    old.mkdir()
    sentinel = old/'retained'
    sentinel.write_bytes(b'older paid state')
    final = tmp_path/'checkpoint-000002'
    final.mkdir()
    leaf = final/'000000.npy'
    leaf.write_bytes(b'opaque numeric checkpoint leaf')
    manifest = {'schema':'xmimic.training-state/v1','iteration':2,
                'leaves':[{'file':leaf.name,'sha256':digest(leaf),'size':leaf.stat().st_size}]}
    (final/'manifest.json').write_text(json.dumps(manifest))
    receipt = {'completed':True,'checkpoint':str(final),'iterations':2}
    selected = checkpoint_files(tmp_path,receipt)
    assert set(selected) == {'checkpoint-000002/manifest.json','checkpoint-000002/000000.npy'}
    assert sentinel.read_bytes() == b'older paid state'
    leaf.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='leaf bytes changed'):
        checkpoint_files(tmp_path,receipt)
    assert sentinel.read_bytes() == b'older paid state'


def test_checkpoint_receipt_cannot_escape_cell(tmp_path):
    with pytest.raises(ValueError,match='final full checkpoint'):
        checkpoint_files(tmp_path,{'completed':True,'iterations':2,'checkpoint':str(Path('/outside'))})


def test_archive_seal_preserves_competing_bytes_and_recovers_verified_staging(tmp_path, monkeypatch):
    source = tmp_path/'source'
    source.write_bytes(b'paid evidence')
    archive = tmp_path/'evidence.tar.gz'
    link = os.link
    def competing_writer(staging, destination):
        destination.write_bytes(b'other retained evidence')
        return link(staging, destination)
    monkeypatch.setattr(os, 'link', competing_writer)
    with pytest.raises((ValueError, tarfile.ReadError)):
        archive_files({'source':source}, archive)
    assert archive.read_bytes() == b'other retained evidence'
    assert archive.with_suffix('.partial').is_file()

    recovered = tmp_path/'recovered.tar.gz'
    def interrupted_seal(*args):
        raise OSError('injected interruption before seal')
    monkeypatch.setattr(os, 'link', interrupted_seal)
    with pytest.raises(OSError, match='injected'):
        archive_files({'source':source}, recovered)
    monkeypatch.setattr(os, 'link', link)
    inventory = archive_files({'source':source}, recovered)
    verify_archive(recovered, inventory)
    assert not recovered.with_suffix('.partial').exists()


def screening_fixture(root):
    screen, prepared, owner, output = (root/name for name in ('screen','prepared','owner','output'))
    for path in (screen/'checkout', screen/'source', root/'review', prepared/'robot', owner/'sx_embodiments'):
        path.mkdir(parents=True)
    def write(path, value):
        path.write_text(json.dumps(value))
    for name in ('embodiment.json','robot/embodiment.json'):
        write(prepared/name, {})
    (prepared/'robot/scene.xml').write_text('<mujoco/>')
    (prepared/'robot/canonical.urdf').write_text('<robot/>')
    (prepared/'mesh.stl').write_bytes(b'mesh')
    write(prepared/'robot/receipt.json', {'files':[]})
    write(prepared/'benchmark.json', dict(embodiment_id='synthetic_fixture',
        scene_sha256=digest(prepared/'robot/scene.xml'), urdf_sha256=digest(prepared/'robot/canonical.urdf'),
        owner_receipt_sha256=digest(prepared/'robot/receipt.json'),
        assets=[dict(path='mesh.stl',sha256=digest(prepared/'mesh.stl'),size_bytes=4)]))
    request = dict(methods=['dls'],seeds=[0],evaluation_seeds=[7], task=dict(episode_steps=1),
        body_sha256=digest(prepared/'embodiment.json'),preparation_sha256=digest(prepared/'benchmark.json'))
    source_files = {}
    for field, name in (('runner_sha256','tools/benchmark_urr.py'),
                        ('environment_sha256','xmimic/urr_benchmark.py'),
                        ('learner_sha256','xmimic/urr_learning.py')):
        source = screen/'checkout'/name
        source.parent.mkdir(exist_ok=True)
        source.write_text(name)
        source_files[name] = request[field] = digest(source)
    write(screen/'request.json', request)
    evaluation = screen/'dls/seed-0/evaluation'
    evaluation.mkdir(parents=True)
    evaluation_request = dict(seeds=[7],environment=dict(body_document_sha256=request['body_sha256'],
        preparation_sha256=request['preparation_sha256'],scene_sha256=digest(prepared/'robot/scene.xml'),
        environment_code_sha256=request['environment_sha256']))
    summary = dict(request_sha256=content_hash(evaluation_request))
    write(evaluation/'request.json', evaluation_request)
    write(evaluation/'summary.json', summary)
    write(evaluation/'task-7.json', dict(seed=7, request_sha256=content_hash(evaluation_request),measurements=[{}]))
    write(root/'review/replay.json',dict(body_id='synthetic_fixture',scene_sha256=digest(prepared/'robot/scene.xml'),
        tasks=[dict(label='DLS',sha256=digest(evaluation/'task-7.json'))]))
    (root/'review/replay.mp4').write_bytes(b'recorded fixture video bytes')
    write(screen/'result.json',dict(complete=True,request_sha256=content_hash(request),
        records=[dict(method='dls',training_seed=0,evaluation=summary)]))
    write(screen/'screening-report.json', dict(run_result_sha256=digest(screen/'result.json')))
    write(screen/'checkout-receipt.json',dict(files=source_files,revision='a'*40))
    for name in ('README.md','LICENSE','THIRD_PARTY_NOTICES.md','pyproject.toml'):
        (owner/name).write_text(name)
    (root/'commit.txt').write_text('b'*40)
    return screen, prepared, owner, output


@pytest.mark.parametrize('mutation', ['request_link','report_link','missing_cell','duplicate_cell','scene','asset','task_request','task_samples'])
def test_preparation_refuses_broken_run_closure_before_archiving(tmp_path, mutation):
    screen, prepared, owner, output = screening_fixture(tmp_path)
    result = json.loads((screen/'result.json').read_bytes())
    if mutation == 'request_link':
        result['request_sha256'] = '0'*64
    elif mutation == 'missing_cell':
        result['records'] = []
    elif mutation == 'duplicate_cell':
        result['records'] *= 2
    if mutation in ('request_link','missing_cell','duplicate_cell'):
        (screen/'result.json').write_text(json.dumps(result))
        (screen/'screening-report.json').write_text(json.dumps(dict(run_result_sha256=digest(screen/'result.json'))))
    elif mutation == 'report_link':
        (screen/'screening-report.json').write_text(json.dumps(dict(run_result_sha256='0'*64)))
    elif mutation in ('scene','asset'):
        (prepared/('robot/scene.xml' if mutation == 'scene' else 'mesh.stl')).write_bytes(b'changed')
    else:
        path = screen/'dls/seed-0/evaluation/task-7.json'
        task = json.loads(path.read_bytes())
        task['request_sha256' if mutation == 'task_request' else 'measurements'] = '0'*64 if mutation == 'task_request' else []
        path.write_text(json.dumps(task))
    with pytest.raises(ValueError):
        prepare(screen,prepared,owner,output,visual_reviews=[tmp_path/'review'])
    assert not list(output.glob('*.tar.gz'))


def test_complete_preparation_reopens_without_changing_evidence(tmp_path):
    args = screening_fixture(tmp_path)
    named, inventory = prepare(*args,visual_reviews=[tmp_path/'review'])
    before = {name:digest(path) for name,path in named.items()}
    reopened, repeated = prepare(*args,visual_reviews=[tmp_path/'review'])
    assert inventory == repeated
    assert before == {name:digest(path) for name,path in reopened.items()}


@pytest.mark.parametrize('source_present', [False, True])
def test_explicit_visuals_and_optional_source_preserve_canonical_checkout(tmp_path, source_present):
    args = screening_fixture(tmp_path)
    screen, _, _, output = args
    source = screen/'source'
    if source_present:
        (source/'snapshot.py').write_bytes(b'optional supplementary source')
    else:
        source.rmdir()  # empty derived fixture only
    second = tmp_path/'comparison'
    second.mkdir()
    (second/'replay.json').write_bytes((tmp_path/'review/replay.json').read_bytes())
    (second/'replay.mp4').write_bytes(b'another recorded view')
    named, inventory = prepare(*args,visual_reviews=[tmp_path/'review',second])
    assert inventory['schema'] == 'xmimic.urr-evidence-inventory/v2'
    import tools.publish_urr_benchmark as publisher
    assert inventory['publisher_sha256'] == digest(Path(publisher.__file__))
    assert 'visual-review.tar.gz' in named and 'visual-review-task-0-v3.tar.gz' not in named
    assert set(inventory['archives']['visual-review.tar.gz']) == {
        'review/replay.json','review/replay.mp4','comparison/replay.json','comparison/replay.mp4'}
    assert ('source/snapshot.py' in inventory['archives']['screening.tar.gz']) == source_present
    assert 'tools/benchmark_urr.py' in inventory['archives']['historical-checkout.tar.gz']
    manifest = json.loads((output/'artifact-manifest.json').read_bytes())
    assert manifest['purpose'] == 'evaluation' and manifest['training_eligible'] is False


def test_old_inventory_is_refused_before_any_archive_changes(tmp_path):
    args = screening_fixture(tmp_path)
    output = args[-1]
    output.mkdir()
    old = b'{"schema":"xmimic.urr-evidence-inventory/v1"}'
    (output/'inventory.json').write_bytes(old)
    (output/'visual-review-task-0-v3.tar.gz').write_bytes(b'immutable old archive')
    before = {p.name:p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError,match='version is immutable'):
        prepare(*args,visual_reviews=[tmp_path/'review'])
    assert before == {p.name:p.read_bytes() for p in output.iterdir()}


@pytest.mark.parametrize('invalid', ['duplicate','wrong_task','wrong_scene'])
def test_explicit_replay_cannot_claim_unrelated_evidence(tmp_path, invalid):
    args = screening_fixture(tmp_path)
    review = tmp_path/'review'
    reviews = [review]
    if invalid == 'duplicate':
        reviews *= 2
    else:
        path = review/'replay.json'
        receipt = json.loads(path.read_bytes())
        if invalid == 'wrong_task':
            receipt['tasks'][0]['sha256'] = '0'*64
        else:
            receipt['scene_sha256'] = '0'*64
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        prepare(*args,visual_reviews=reviews)
    assert not list(args[-1].glob('*.tar.gz'))
