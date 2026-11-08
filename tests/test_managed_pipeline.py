"""Local artifact handoff integrity; hosted scheduling requires separate traversal."""

import pytest
from sx_contracts.identity import file_digest
from sx_training.artifacts import ArtifactFile, ArtifactManifest, ArtifactPurpose

from xmimic.managed_pipeline import JobSettings, Pack, training_candidate


def candidate(tmp_path, *, eligible=True, payload=b"whole feasible clip"):
    motion = tmp_path / "motion.npz"
    motion.write_bytes(payload)
    item = ArtifactFile("motion.npz", file_digest(motion), len(payload))
    selection = ArtifactManifest(ArtifactPurpose.TRANSFORM, (item,), eligible)
    path = tmp_path / "training-candidate.json"
    path.write_bytes(selection.canonical_bytes())
    manifest = ArtifactManifest(
        ArtifactPurpose.TRANSFORM,
        (
            item,
            ArtifactFile(path.name, file_digest(path), path.stat().st_size),
        ),
        eligible,
    )
    return Pack("retarget.result", "a" * 64, manifest, tmp_path)


def test_handoff_preserves_exact_files_and_requests_learning_admission(tmp_path):
    pack = candidate(tmp_path)
    selected = training_candidate(pack)
    assert selected.purpose is ArtifactPurpose.TRAINING
    assert selected.files == (pack.manifest.files[0],)
    assert selected.training_eligible


def test_evaluation_only_source_stays_out_of_training(tmp_path):
    with pytest.raises(ValueError, match="not eligible"):
        training_candidate(candidate(tmp_path, eligible=False))


def test_candidate_cannot_smuggle_unpublished_or_changed_bytes(tmp_path):
    pack = candidate(tmp_path)
    (tmp_path / "motion.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="admitted identity"):
        training_candidate(pack)
    (tmp_path / "motion.npz").write_bytes(b"whole feasible clip")
    altered = ArtifactManifest(
        ArtifactPurpose.TRANSFORM,
        (ArtifactFile("new.npz", pack.manifest.files[0].sha256, 19),),
        True,
    )
    (tmp_path / "training-candidate.json").write_bytes(altered.canonical_bytes())
    with pytest.raises(ValueError, match="admitted identity"):
        training_candidate(pack)


@pytest.mark.parametrize(
    "field,value",
    [
        ("image", "unversioned:latest"),
        ("cpus", 0),
        ("timeout_s", -1),
        ("spending_limit_usd_micros", 0),
        ("memory_gib", 1025),
        ("steps", True),
    ],
)
def test_invalid_paid_request_fails_before_submission(field, value):
    args = dict(
        image="repo/image@sha256:" + "a" * 64,
        timeout_s=60,
        spending_limit_usd_micros=100000,
        cpus=4,
        memory_gib=16,
        steps=1,
        config={},
    )
    args[field] = value
    with pytest.raises(ValueError):
        JobSettings(**args)
