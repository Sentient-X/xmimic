"""Independent rigid geometry and clock oracles for nominal reference projection."""

from dataclasses import replace
import hashlib
import json

import mujoco
import numpy as np
import pytest
from sx_contracts.identity import content_digest

from test_embodiment_bundle import fixture
from xmimic.embodiment_bundle import adapt_nominal_bundle
from xmimic.model import load_model
from xmimic.retarget_quality import NominalQuality, geometry_lowest_z, qualify_motion


def bundle(tmp_path):
    training = fixture(tmp_path)
    path = tmp_path / "scene.xml"
    xml = path.read_text().replace(
        "<worldbody>",
        '<default><geom contype="2" conaffinity="1"/></default><worldbody>',
    )
    xml = xml.replace(
        'name="floor"', 'name="nominal_floor" contype="1" conaffinity="2"'
    )
    path.write_text(xml)
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    receipt["profile"]["collision_mode"] = "ground_only"
    receipt["profile_sha256"] = str(content_digest(receipt["profile"]))
    receipt["scene_sha256"] = hashlib.sha256(xml.encode()).hexdigest()
    receipt["files"] = [["scene.xml", receipt["scene_sha256"]]]
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    adapt_nominal_bundle(tmp_path, training)
    return tmp_path


def contract():
    return NominalQuality(0.03, 0.01, ("sole",), 0.002, 1.0, 4.0, 6.0, 0.001)


def motion():
    return dict(
        q=np.array([[0.1, 0.1], [0.2, 0.1], [0.3, 0.1]]),
        root_p=np.array([[1.0, 2.0, 0.6], [1.1, 2.0, 0.7], [1.2, 2.0, 0.8]]),
        root_quat=np.tile([1.0, 0, 0, 0], (3, 1)),
        fps=np.asarray(120.0),
        joint_names=np.array(["twist", "bend"]),
        embodiment_id=np.asarray("fixture-only"),
    )


def test_exact_sphere_sole_grounding_clock_and_preserved_coordinates(tmp_path):
    root = bundle(tmp_path)
    original = motion()
    projected, report = qualify_motion(
        original, np.full((3, 1), 0.02), np.zeros((3, 1)), root, contract()
    )
    assert (
        report["accepted"]
        and not report["physical_qualified"]
        and not report["self_collision_qualified"]
    )
    assert float(projected["fps"]) == 30.0
    assert report["max_joint_speed_rad_s"] == pytest.approx(3.0)
    np.testing.assert_array_equal(projected["q"], original["q"])
    np.testing.assert_array_equal(projected["root_p"][:, :2], original["root_p"][:, :2])
    np.testing.assert_array_equal(original["root_p"][:, 2], [0.6, 0.7, 0.8])
    model, meta = load_model(root)
    data = mujoco.MjData(model)
    for i in range(3):
        data.qpos[:3] = projected["root_p"][i]
        data.qpos[meta["qpos_adr"]] = projected["q"][i]
        mujoco.mj_forward(model, data)
        # Independent sphere oracle, rather than reusing geometry_lowest_z.
        assert data.xpos[model.body("sole").id, 2] - 0.04 == pytest.approx(0.002)


@pytest.mark.parametrize(
    "change,check",
    [
        ({"position_tolerance_m": 0.01}, "landmark_position"),
        ({"max_joint_speed_rad_s": 2.0}, "joint_speed"),
        ({"max_root_shift_m": 0.001}, "root_projection"),
    ],
)
def test_quality_rejection_keeps_all_frames(tmp_path, change, check):
    projected, report = qualify_motion(
        motion(),
        np.full((3, 1), 0.02),
        np.zeros((3, 1)),
        bundle(tmp_path),
        replace(contract(), **change),
    )
    assert not report["accepted"] and not report["checks"][check]
    assert len(projected["q"]) == 3


def test_rotated_box_support_matches_corner_oracle():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="box" size=".2 .3 .4" pos="1 2 3" euler=".2 .4 .8"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    corners = np.array(
        [[a, b, c] for a in [-0.2, 0.2] for b in [-0.3, 0.3] for c in [-0.4, 0.4]]
    )
    expected = (corners @ data.geom_xmat[0].reshape(3, 3).T + data.geom_xpos[0])[
        :, 2
    ].min()
    assert geometry_lowest_z(model, data, 0) == pytest.approx(expected)


def test_invalid_contract_and_bad_identity_refuse(tmp_path):
    with pytest.raises(ValueError):
        replace(contract(), time_scale=0.5)
    with pytest.raises(ValueError):
        replace(contract(), max_root_shift_m=True)
    wrong = motion()
    wrong["embodiment_id"] = np.asarray("another")
    with pytest.raises(ValueError, match="body"):
        qualify_motion(
            wrong, np.zeros((3, 1)), np.zeros((3, 1)), bundle(tmp_path), contract()
        )


def test_retained_requalification_preserves_old_rejection_and_reuses_verified_cache(
    tmp_path, monkeypatch
):
    from dataclasses import asdict
    from test_retarget_workflow import recipe
    from xmimic.artifact_cache import ArtifactCache, CacheIdentity, ClipDisposition
    from xmimic.retarget_workflow import qualify_cached_candidate

    robot = tmp_path / "robot"
    robot.mkdir()
    bundle(robot)
    owner = json.loads((robot / "receipt.json").read_text())
    owner["embodiment_id"] = "a" * 64
    owner["urdf_sha256"] = "b" * 64
    owner["profile"]["expected_embodiment_id"] = "a" * 64
    owner["profile_sha256"] = str(content_digest(owner["profile"]))
    (robot / "receipt.json").write_text(json.dumps(owner))
    meta = json.loads((robot / "meta.json").read_text())
    meta["embodiment_id"] = "a" * 64
    (robot / "meta.json").write_text(json.dumps(meta))
    arrays = motion()
    arrays["embodiment_id"] = np.asarray("a" * 64)
    original_npz = tmp_path / "original.npz"
    np.savez(original_npz, **arrays)
    settings = recipe()
    raw_report = dict(
        recipe=asdict(settings),
        source_sha256="c" * 64,
        source_group="test-family",
        frame_count=3,
        geometric_result=dict(
            position_residual_m=[[0.02]] * 3,
            orientation_residual_rad=[[0.0]] * 3,
            dispositions=["residual_exceeded"] * 3,
        ),
    )
    report = tmp_path / "report.json"
    report.write_text(json.dumps(raw_report))
    identity = CacheIdentity(
        "c" * 64, "d" * 64, "a" * 64, "b" * 64, "e" * 64, "f" * 64, "0" * 64
    )
    source_cache = ArtifactCache(tmp_path / "source-cache")
    with source_cache.entry(identity, ["clip"]) as entry:
        entry.start_clip("clip", "old-solver")
        entry.settle_clip(
            "clip",
            "old-solver",
            status=ClipDisposition.REJECTED,
            artifacts={"candidate.npz": original_npz, "retarget.json": report},
            metadata={"training_admitted": False},
            reason="old 2mm gate",
        )
        entry.complete()
        source_dir = entry.path
    before = (source_dir / "manifest.json").read_bytes()
    target = ArtifactCache(tmp_path / "derived")
    settings = replace(settings, nominal_quality=contract())
    result = qualify_cached_candidate(
        source_dir, settings, robot, cache=target, execution_ref="new-contract"
    )
    assert result.disposition is ClipDisposition.SUCCEEDED
    assert (source_dir / "manifest.json").read_bytes() == before
    assert json.loads(before)["clips"]["clip"]["status"] == "rejected"
    monkeypatch.setattr(
        "xmimic.retarget_workflow.qualify_motion",
        lambda *a, **k: pytest.fail("must reuse paid derived work"),
    )
    assert qualify_cached_candidate(
        source_dir, settings, robot, cache=target, execution_ref="new-contract"
    ).cache_hit
    with pytest.raises(ValueError, match="mapping or optimizer"):
        qualify_cached_candidate(
            source_dir,
            replace(settings, root_translation_scale=2),
            robot,
            cache=target,
            execution_ref="changed",
        )


def test_direction_only_nominal_quality_reaches_explicit_direction_gate(tmp_path):
    from test_retarget_workflow import recipe
    from xmimic.retarget_target import Direction, TargetProfile
    from xmimic.retarget_workflow import _direction_quality
    arrays, report = qualify_motion(motion(), np.zeros((3, 0)), np.zeros((3, 0)), bundle(tmp_path), contract())
    assert report['max_position_residual_m'] is None
    assert report['max_orientation_residual_rad'] is None
    directional = replace(recipe(), target=TargetProfile((), directions=(Direction('A', 'B', 'a', 'b'),)),
                          landmark_scales=(), orientation_offsets=())
    _direction_quality(report, np.full((3, 1), .5), directional, len(arrays['q']))
    assert not report['accepted'] and not report['checks']['direction_residual']
