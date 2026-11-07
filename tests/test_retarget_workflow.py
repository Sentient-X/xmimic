"""Whole source-parse/FK/IK/cache journey against independent planar geometry."""
from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest
from sx_contracts.identity import canonical_json

from test_retarget_target import fixture_body, profile
from xmimic.artifact_cache import ArtifactCache, CacheError, ClipDisposition
from xmimic.retarget_target import IKConfig
from xmimic.retarget_workflow import RetargetRecipe, recipe_from_json, retarget_clip


def recipe():
    return RetargetRecipe(profile(), tuple(map(tuple, np.eye(4))), 'Root', tuple(map(tuple, np.eye(3))),
                          (0., 0., 0.), 1., (('hand', 1.),), (('hand', tuple(map(tuple, np.eye(3)))),),
                          (('q1', .2), ('q2', .4), ('q3', .3)),
                          IKConfig(position_tolerance_m=1e-6, orientation_tolerance_rad=1e-6))


def source(unreachable=False):
    rows = []
    q1, q2 = np.linspace(.2, .5, 8), np.linspace(.4, .6, 8)
    for i, (a, b) in enumerate(zip(q1, q2)):
        yaw = -.3 + .5*i/7
        x = math.cos(a) + 1.2*math.cos(a+b)
        y = math.sin(a) + 1.2*math.sin(a+b)
        if unreachable and i == 2:
            x += 100
        rows.append(' '.join(map(str, [10+i, 0, 50+i, math.degrees(yaw), 0, 0,
                                       100*x, 100*y, 0, math.degrees(a+b), 0, 0])))
    header = '''HIERARCHY
ROOT Root
{
 OFFSET 0 0 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT hand
 {
  OFFSET 999 999 999
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 }
}
MOTION
Frames: 8
Frame Time: 0.033333333333333333
'''
    return (header+'\n'.join(rows)+'\n').encode(), q1, q2


def run(tmp_path, monkeypatch, *, data=None, settings=None, shape=None):
    body = fixture_body(tmp_path, monkeypatch)
    result = retarget_clip(body, source()[0] if data is None else data, recipe() if settings is None else settings,
                           expected_embodiment_id=str(body.id), cache=ArtifactCache(tmp_path/'cache'),
                           clip_id='synthetic-clip', source_group='synthetic-family', execution_ref='cpu-test-attempt',
                           shape_sha256=shape)
    manifest = json.loads((result.directory/'manifest.json').read_text())
    return result, manifest


def test_whole_clip_known_angles_and_cache_hit_without_solver(tmp_path, monkeypatch):
    result, document = run(tmp_path, monkeypatch)
    assert result.disposition is ClipDisposition.SUCCEEDED and not result.cache_hit
    artifacts = document['clips']['synthetic-clip']['artifacts']
    with np.load(result.directory/'blobs'/artifacts['motion.npz']['sha256'], allow_pickle=False) as motion:
        _, q1, q2 = source()
        np.testing.assert_allclose(motion['q'][:,0], q2, atol=1e-5)
        np.testing.assert_allclose(motion['q'][:,1], q1, atol=1e-5)
        np.testing.assert_array_equal(motion['q'][:,2], .3)
        assert motion['q'].shape == (8,3)
        np.testing.assert_allclose(motion['root_p'][:,0], (10+np.arange(8))*.01)
    def forbidden(*args, **kwargs):
        raise AssertionError('cached work must not execute again')
    monkeypatch.setattr('xmimic.retarget_workflow.solve_sequence', forbidden)
    reused, _ = run(tmp_path, monkeypatch)
    assert reused.cache_hit and reused.manifest_sha256 == result.manifest_sha256


def test_rejection_retains_all_frames_and_residuals_but_no_training_motion(tmp_path, monkeypatch):
    result, document = run(tmp_path, monkeypatch, data=source(True)[0])
    assert result.disposition is ClipDisposition.REJECTED
    clip = document['clips']['synthetic-clip']
    assert set(clip['artifacts']) == {'candidate.npz', 'retarget.json'}
    assert clip['metadata']['training_admitted'] is False
    report = json.loads((result.directory/'blobs'/clip['artifacts']['retarget.json']['sha256']).read_text())
    assert report['frame_count'] == 8 and report['training_admitted'] is False
    with np.load(result.directory/'blobs'/clip['artifacts']['candidate.npz']['sha256'], allow_pickle=False) as candidate:
        assert candidate['q'].shape == (8,3)
    reused, _ = run(tmp_path, monkeypatch, data=source(True)[0])
    assert reused.cache_hit and reused.disposition is ClipDisposition.REJECTED


def test_source_shape_and_recipe_mutations_invalidate_cache(tmp_path, monkeypatch):
    first, _ = run(tmp_path, monkeypatch)
    changed_bytes, _ = run(tmp_path, monkeypatch, data=source()[0]+b'\n')
    changed_shape, _ = run(tmp_path, monkeypatch, shape='a'*64)
    changed_recipe, _ = run(tmp_path, monkeypatch, settings=replace(recipe(), solver=IKConfig()))
    assert len({r.cache_key for r in (first, changed_bytes, changed_shape, changed_recipe)}) == 4
    assert not any(r.cache_hit for r in (changed_bytes, changed_shape, changed_recipe))


def test_cache_output_mutation_is_refused_without_recompute(tmp_path, monkeypatch):
    result, document = run(tmp_path, monkeypatch)
    artifact = document['clips']['synthetic-clip']['artifacts']['motion.npz']['sha256']
    (result.directory/'blobs'/artifact).write_bytes(b'corruption')
    with pytest.raises(CacheError, match='disagree'):
        run(tmp_path, monkeypatch)


def test_recipe_roundtrip_and_strict_json_boundary():
    encoded = canonical_json(asdict(recipe())).encode()
    assert recipe_from_json(encoded) == recipe()
    for bad in [encoded[:-1]+b',"source_root":"Root"}',
                encoded.replace(b'"root_translation_scale":1.0', b'"root_translation_scale":true'),
                encoded.replace(b'"root_translation_scale":1.0', b'"root_translation_scale":"1"'),
                encoded.replace(b'"root_translation_scale":1.0', b'"root_translation_scale":NaN')]:
        with pytest.raises(ValueError):
            recipe_from_json(bad)
    raw = json.loads(encoded)
    raw['world_from_source'][0][0] = -1
    with pytest.raises(ValueError):
        recipe_from_json(json.dumps(raw).encode())


def test_invalid_seed_is_refused_before_retained_solver_attempt(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='held'):
        run(tmp_path, monkeypatch, settings=replace(recipe(), initial_q=(('q1',.2),('q2',.4),('q3',.2))))
    assert not list((tmp_path/'cache').glob('*/state.json'))


def segmented_source(zero=False):
    data, q1, q2 = source()
    header, rows = data.decode().split('Frame Time: 0.033333333333333333\n')
    header = header.replace('\n }\n}\nMOTION', '\n }\n JOINT elbow\n {\n OFFSET 0 0 0\n CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n }\n}\nMOTION')
    changed = []
    for line, angle in zip(rows.splitlines(), q1):
        values = list(map(float,line.split()))
        values[6:9] = [v*10 for v in values[6:9]]  # Human is ten times the target size.
        elbow = [0.,0.,0.] if zero else [1000*math.cos(angle),1000*math.sin(angle),0.]
        changed.append(' '.join(map(str,values+elbow+[0.,0.,0.])))
    return (header+'Frame Time: 0.033333333333333333\n'+'\n'.join(changed)+'\n').encode()


def test_anatomical_segments_recover_target_angles_despite_different_source_lengths(tmp_path,monkeypatch):
    from xmimic.retarget_workflow import SegmentTarget
    settings = replace(recipe(),landmark_scales=(),segment_targets=(
        SegmentTarget('hand',(0.,0.,0.),(('Root','elbow',1.),('elbow','hand',1.2))),))
    assert recipe_from_json(canonical_json(asdict(settings)).encode()) == settings
    result, document = run(tmp_path,monkeypatch,data=segmented_source(),settings=settings)
    assert result.disposition is ClipDisposition.SUCCEEDED
    artifact = document['clips']['synthetic-clip']['artifacts']['motion.npz']['sha256']
    with np.load(result.directory/'blobs'/artifact,allow_pickle=False) as motion:
        _, q1, q2 = source()
        np.testing.assert_allclose(motion['q'][:,0],q2,atol=1e-5)
        np.testing.assert_allclose(motion['q'][:,1],q1,atol=1e-5)


def test_zero_length_anatomical_segment_refuses_before_solver_start(tmp_path,monkeypatch):
    from xmimic.retarget_workflow import SegmentTarget
    settings = replace(recipe(),landmark_scales=(),segment_targets=(
        SegmentTarget('hand',(0.,0.,0.),(('Root','elbow',1.),('elbow','hand',1.2))),))
    with pytest.raises(ValueError,match='zero-length'):
        run(tmp_path,monkeypatch,data=segmented_source(zero=True),settings=settings)
    states = list((tmp_path/'cache').glob('*/state.json'))
    assert len(states) == 1
    assert json.loads(states[0].read_text())['document']['clips']['synthetic-clip']['status'] == 'pending'
