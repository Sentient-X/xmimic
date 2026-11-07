"""Generic preparation keeps canonical body/source identities and validates before copying."""
from dataclasses import replace
import json

import pytest
pytest.importorskip('sx_training.artifacts')
from sx_contracts.identity import canonical_json
from sx_embodiments import Embodiment

from test_managed_worker import context_fixture
from tools.prepare_embodiment_job import prepare, training_profile


def inputs(context):
    artifacts = context.artifacts
    body = Embodiment.from_json(artifacts['embodiment.json'].read_text())
    nominal = json.loads(artifacts['robot/receipt.json'].read_text())['profile']
    return body, artifacts['source.bvh'].read_bytes(), canonical_json(nominal).encode(), artifacts['recipe.json'].read_bytes(), artifacts['training-profile.json'].read_bytes()


def test_generic_prepare_emits_complete_portable_closure(tmp_path,monkeypatch):
    context = context_fixture(tmp_path,monkeypatch)
    args = inputs(context)
    output = tmp_path/'prepared'
    report = prepare(*args,source_entry='private-source-parent',output=output)
    assert report['embodiment_id'] == str(args[0].id)
    assert report['training_admitted'] is False
    assert report['source_frames'] == 8
    assert (output/'assets/robot.urdf').read_bytes() == args[0].urdf_bytes
    assert (output/'source.bvh').read_bytes() == args[1]
    assert (output/'robot/meta.json').is_file()
    assert json.loads((output/'robot/receipt.json').read_text())['embodiment_id'] == str(args[0].id)
    with pytest.raises(ValueError,match='new derived'):
        prepare(*args,source_entry='private-source-parent',output=output)


def test_missing_segment_joint_refused_before_materialization(tmp_path,monkeypatch):
    from xmimic.retarget_workflow import SegmentTarget,recipe_from_json
    from dataclasses import asdict
    context = context_fixture(tmp_path,monkeypatch)
    body,source,nominal,recipe_bytes,training = inputs(context)
    recipe = replace(recipe_from_json(recipe_bytes),landmark_scales=(),segment_targets=(
        SegmentTarget('hand',(0.,0.,0.),(('Root','absent',1.),)),))
    with pytest.raises(ValueError,match='absent'):
        prepare(body,source,nominal,canonical_json(asdict(recipe)).encode(),training,
                source_entry='private-source-parent',output=tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_training_profile_parser_refuses_duplicate_and_unknown_fields(tmp_path,monkeypatch):
    context = context_fixture(tmp_path,monkeypatch)
    payload=context.artifacts['training-profile.json'].read_bytes()
    for bad in (payload[:-1]+b',"njmax":1}',payload[:-1]+b',"other":1}'):
        with pytest.raises(ValueError):
            training_profile(bad)
