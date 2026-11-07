"""Actual sx_training context -> CPU authored program -> immutable candidate closure."""
from dataclasses import asdict, replace
import hashlib
import json

import pytest

pytest.importorskip('sx_training.artifacts')
from sx_contracts import Sha256Digest
from sx_contracts.assets import AssetFormat, AssetRole, AssetRef
from sx_contracts.identity import canonical_json
from sx_embodiments.nominal_mujoco import NominalJoint, NominalProfile, SupportContact, export_nominal_mujoco
from sx_training import TrainingContext, TrainerInvocation, execution_bytes
from sx_training.artifacts import ArtifactFile, ArtifactManifest, ArtifactPurpose, artifact_manifest_from_bytes

from test_retarget_target import fixture_body
from test_retarget_workflow import recipe, source
from xmimic.embodiment_bundle import TrainingProfile
from xmimic.managed_worker import retarget, train


def context_fixture(tmp_path, monkeypatch, rejected=False):
    assets = tmp_path / 'data' / 'inputs' / 'assets'
    assets.mkdir(parents=True)
    body = fixture_body(assets, monkeypatch)
    original = body.urdf_bytes
    # A real owner export with explicit inertials and simple collision geometry.
    xml = original
    for name in ('base','a','b','tip','held','rod'):
        xml = xml.replace(f'<link name="{name}"/>'.encode(),
                          f'<link name="{name}"><inertial><mass value=".1"/><inertia ixx=".001" iyy=".001" izz=".001" ixy="0" ixz="0" iyz="0"/></inertial><collision><geometry><sphere radius=".02"/></geometry></collision></link>'.encode())
    (assets/'robot.urdf').write_bytes(xml)
    asset = replace(body.assets[0], asset=AssetRef.from_bytes(xml, uri='package://sx-embodiments/robot.urdf', format=AssetFormat.URDF, role=AssetRole.DESCRIPTION))
    body = replace(body, assets=(asset,))
    root = assets.parent
    (root/'embodiment.json').write_text(body.to_json())
    (root/'source.bvh').write_bytes(source(rejected)[0])
    (root/'recipe.json').write_text(canonical_json(asdict(recipe())))
    nominal = NominalProfile(str(body.id), tuple(NominalJoint(n,10.,1.,10.,.01,0.,.2) for n in body.joint_names),
                             (), ('tip',), (SupportContact('tip',.02),), (0.,0.,.5),(1.,0.,0.,0.),.005,
                             (0.,0.,-9.81),(1.,.005,.0001),('Synthetic analytic test only',))
    export_nominal_mujoco(body, nominal, root/'robot')
    profile = TrainingProfile('base', ((0.,0.),)*3,(.9,1.1),0.,(0.,)*3,(0.,)*3,(0.,)*3,(0.,)*3,0.,0.,64,32)
    (root/'training-profile.json').write_text(canonical_json(asdict(profile)))
    manifest = ArtifactManifest(ArtifactPurpose.TRANSFORM, tuple(
        ArtifactFile(p.relative_to(root).as_posix(),Sha256Digest(hashlib.sha256(p.read_bytes()).hexdigest()),p.stat().st_size)
        for p in sorted(root.rglob('*')) if p.is_file()), False)
    config = {'embodiment_id':str(body.id),'clip_id':'whole-clip','source_group':'fixture-family','shape_sha256':None}
    doc = execution_bytes({'job_id':'ftjob_'+'a'*32,'spec':{'artifact_inputs':{'manifest_json':manifest.canonical_bytes().decode()}}},
                           TrainerInvocation('xmimic.managed_worker:retarget',config))
    ctx = TrainingContext(doc,data_root=tmp_path/'data',cache_dir=tmp_path/'persistent-cache',runs_dir=tmp_path/'runs')
    return ctx


def test_actual_managed_cpu_program_preserves_eligibility_and_reuses_cache(tmp_path,monkeypatch):
    context = context_fixture(tmp_path,monkeypatch)
    output = retarget(context)
    manifest = artifact_manifest_from_bytes((output/'training-candidate.json').read_bytes())
    assert manifest.purpose is ArtifactPurpose.TRANSFORM
    assert manifest.training_eligible is False
    assert {'motion/clip.npz','robot/scene.xml','robot/meta.json','recipe.json','training-profile.json'} <= {f.path for f in manifest.files}
    assert not any('candidate.npz' in f.path for f in manifest.files)
    before = (output/'training-candidate.json').read_bytes()
    def forbidden(*args,**kwargs):
        raise AssertionError('managed retry bought the solver twice')
    monkeypatch.setattr('xmimic.retarget_workflow.solve_sequence',forbidden)
    assert retarget(context) == output
    assert (output/'training-candidate.json').read_bytes() == before


def test_actual_managed_rejection_has_diagnostics_and_no_training_candidate(tmp_path,monkeypatch):
    context = context_fixture(tmp_path,monkeypatch,rejected=True)
    output = retarget(context)
    assert not (output/'training-candidate.json').exists()
    assert not (output/'motion').exists()
    assert (output/'diagnostics'/'candidate.npz').is_file()
    assert json.loads((output/'candidate.json').read_text())['feasible'] is False


def test_gpu_program_refuses_transform_inputs_before_any_gpu_initialization(tmp_path,monkeypatch):
    context = context_fixture(tmp_path,monkeypatch)
    context.trainer = TrainerInvocation('xmimic.managed_worker:train',{'training':{},'source_binding':'a'*64,'embodiment_id':'b'*64})
    with pytest.raises(ValueError,match='eligible training'):
        train(context)


@pytest.mark.parametrize('mutation', ['none','bytes','order','missing','whole'])
def test_final_recovery_chunks_cross_boundaries_and_verify_identity(tmp_path, mutation):
    from xmimic.managed_worker import export_recovery_archive, restore_recovery_archive
    archive = tmp_path/'original.tar'
    # Nonperiodic content catches byte order and off-by-one errors at three boundaries.
    content = bytes((i*i + 17*i + i//7) % 256 for i in range(1031))
    archive.write_bytes(content)
    parts = tmp_path/'parts'
    export_recovery_archive(archive, parts, chunk_bytes=257)
    index = json.loads((parts/'index.json').read_text())
    assert [c['size_bytes'] for c in index['chunks']] == [257,257,257,257,3]
    if mutation == 'bytes':
        (parts/'part-000001.bin').write_bytes(b'x'*257)
    elif mutation == 'order':
        index['chunks'][1], index['chunks'][2] = index['chunks'][2], index['chunks'][1]
    elif mutation == 'missing':
        (parts/'part-000002.bin').unlink()  # disposable derived test output
    elif mutation == 'whole':
        index['sha256'] = '0'*64
    (parts/'index.json').write_text(canonical_json(index))
    if mutation == 'none':
        restored = restore_recovery_archive(parts,tmp_path/'restored.tar')
        assert restored.read_bytes() == content
        assert restore_recovery_archive(parts,restored) == restored
        assert export_recovery_archive(archive,parts,chunk_bytes=257) == parts/'index.json'
    else:
        with pytest.raises(ValueError):
            restore_recovery_archive(parts,tmp_path/'restored.tar')
        assert not (tmp_path/'restored.tar').exists()
