"""File/identity/restart proof; these tests do not establish remote Catalog integration."""

import hashlib
import json
import multiprocessing
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from xmimic.artifact_cache import (
    AmbiguousExecution,
    ArtifactCache,
    CacheError,
    CacheIdentity,
    CatalogPublisher,
    Provenance,
    content_hash,
)

PROJECT = "b98ba48a-39c1-4346-98a4-d9c130144a42"
PROVENANCE = Provenance(
    "https://example.test/retarget",
    "commit-pinned",
    "LicenseRef-Test",
    "retarget-program@sha256:code",
)


def identity():
    return CacheIdentity(
        *(
            hashlib.sha256(n.encode()).hexdigest()
            for n in (
                "sources",
                "shape",
                "body",
                "urdf",
                "mapper",
                "solver",
                "settings",
            )
        )
    )


def output(tmp_path, content=b"retargeted clip bytes"):
    path = tmp_path / "output.npz"
    path.write_bytes(content)
    return path


def finish(entry, path):
    entry.start_clip("walk", "local:one")
    entry.settle_clip(
        "walk",
        "local:one",
        status="succeeded",
        artifacts={"motion.npz": path},
        metadata={"frames": 4, "qualified": False},
    )
    return entry.complete()


@pytest.mark.parametrize("field", list(CacheIdentity.__dataclass_fields__))
def test_every_input_fact_changes_cache_key(field):
    original = identity()
    assert replace(original, **{field: "a" * 64}).key != original.key
    with pytest.raises(CacheError):
        replace(original, **{field: "unversioned"})


def test_settings_hash_canonical_and_finite():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    with pytest.raises(CacheError):
        content_hash({"epsilon": float("nan")})


def test_complete_roundtrip_keeps_every_disposition(tmp_path):
    cache = ArtifactCache(tmp_path / "cache")
    inventory = ["walk", "unsupported", "bad"]
    with cache.entry(identity(), inventory) as entry:
        entry.start_clip("walk", "cpu:123")
        entry.settle_clip(
            "walk",
            "cpu:123",
            status="succeeded",
            artifacts={"motion.npz": output(tmp_path)},
        )
        for clip, state in [("unsupported", "rejected"), ("bad", "failed")]:
            entry.start_clip(clip, "cpu:" + clip)
            entry.settle_clip(
                clip, "cpu:" + clip, status=state, reason="retained explanation"
            )
        manifest_sha = entry.complete()
        assert entry.complete() == manifest_sha
    with cache.entry(identity(), list(reversed(inventory))) as entry:
        assert entry.document["complete"]
        assert {c["status"] for c in entry.document["clips"].values()} == {
            "succeeded",
            "failed",
            "rejected",
        }
        assert len(entry.files()) == 2
        with pytest.raises(CacheError, match="immutable"):
            entry.start_clip("bad", "cpu:retry")


def test_restart_never_reexecutes_ambiguous_paid_job(tmp_path):
    cache = ArtifactCache(tmp_path / "cache")
    with cache.entry(identity(), ["walk"]) as entry:
        entry.start_clip("walk", "job:retained-paid-operation")
    with cache.entry(identity(), ["walk"]) as entry:
        with pytest.raises(AmbiguousExecution):
            entry.start_clip("walk", "job:new")
        with pytest.raises(CacheError, match="original"):
            entry.settle_clip(
                "walk", "job:new", status="failed", reason="wrong operation"
            )
        entry.settle_clip(
            "walk",
            "job:retained-paid-operation",
            status="failed",
            reason="original operation settled failed",
        )
        entry.complete()


def test_changed_inventory_and_incomplete_publication_refused(tmp_path):
    cache = ArtifactCache(tmp_path)
    with cache.entry(identity(), ["walk"]) as entry:
        with pytest.raises(CacheError, match="Every clip"):
            entry.complete()
        with pytest.raises(CacheError, match="completed"):
            entry.files()
    with pytest.raises(CacheError, match="inventory"):
        with cache.entry(identity(), ["other"]):
            pass


@pytest.mark.parametrize("target", ["state.json", "manifest.json", "blob"])
def test_corruption_refused_without_repair_or_loss(tmp_path, target):
    cache = ArtifactCache(tmp_path / "cache")
    with cache.entry(identity(), ["walk"]) as entry:
        finish(entry, output(tmp_path))
        corrupt = (
            next(entry.blobs.iterdir()) if target == "blob" else entry.path / target
        )
    corrupt.write_bytes(b"corrupt retained evidence")
    with pytest.raises((CacheError, ValueError)):
        with cache.entry(identity(), ["walk"]):
            pass
    assert corrupt.read_bytes() == b"corrupt retained evidence"


def test_missing_state_does_not_authorize_new_work(tmp_path):
    cache = ArtifactCache(tmp_path)
    with cache.entry(identity(), ["walk"]) as entry:
        entry.start_clip("walk", "paid:123")
        state = entry.state_path
    state.unlink()
    with pytest.raises(CacheError, match="lost its state"):
        with cache.entry(identity(), ["walk"]):
            pass


def test_entry_cannot_mutate_after_releasing_lock(tmp_path):
    with ArtifactCache(tmp_path).entry(identity(), ["walk"]) as entry:
        pass
    with pytest.raises(CacheError, match="lock"):
        entry.start_clip("walk", "later")


def test_crash_after_blob_before_state_keeps_running_and_reconciles(
    tmp_path, monkeypatch
):
    cache = ArtifactCache(tmp_path / "cache")
    source = output(tmp_path)
    with cache.entry(identity(), ["walk"]) as entry:
        entry.start_clip("walk", "local:original")

        def crash():
            raise OSError("simulated interrupted state commit")

        monkeypatch.setattr(entry, "_save", crash)
        with pytest.raises(OSError):
            entry.settle_clip(
                "walk",
                "local:original",
                status="succeeded",
                artifacts={"motion.npz": source},
            )
    with cache.entry(identity(), ["walk"]) as entry:
        assert entry.document["clips"]["walk"]["status"] == "running"
        assert len(list(entry.blobs.iterdir())) == 1
        entry.settle_clip(
            "walk",
            "local:original",
            status="succeeded",
            artifacts={"motion.npz": source},
        )
        entry.complete()


def test_crash_after_manifest_before_state_can_finalize_same_bytes(
    tmp_path, monkeypatch
):
    cache = ArtifactCache(tmp_path / "cache")
    with cache.entry(identity(), ["walk"]) as entry:
        entry.start_clip("walk", "local:1")
        entry.settle_clip("walk", "local:1", status="failed", reason="retained")
        monkeypatch.setattr(
            entry, "_save", lambda: (_ for _ in ()).throw(OSError("crash"))
        )
        with pytest.raises(OSError):
            entry.complete()
        manifest = (entry.path / "manifest.json").read_bytes()
    with cache.entry(identity(), ["walk"]) as entry:
        entry.complete()
        assert (entry.path / "manifest.json").read_bytes() == manifest


def _lock_contender(root, ready, acquired):
    ready.set()
    with ArtifactCache(Path(root)).entry(identity(), ["walk"]):
        acquired.set()


def test_process_lock_excludes_another_writer(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready, acquired = context.Event(), context.Event()
    with ArtifactCache(tmp_path).entry(identity(), ["walk"]):
        process = context.Process(
            target=_lock_contender, args=(str(tmp_path), ready, acquired)
        )
        process.start()
        assert ready.wait(5)
        assert not acquired.wait(0.2)
    assert acquired.wait(5)
    process.join(5)
    assert process.exitcode == 0


@pytest.mark.parametrize("logical", ["../escape", "/absolute", "a/../b", "a\\b"])
def test_unsafe_artifact_names_refused(tmp_path, logical):
    with ArtifactCache(tmp_path / "cache").entry(identity(), ["walk"]) as entry:
        entry.start_clip("walk", "local:1")
        with pytest.raises(CacheError, match="path"):
            entry.settle_clip(
                "walk",
                "local:1",
                status="succeeded",
                artifacts={logical: output(tmp_path)},
            )


def test_symlink_output_refused(tmp_path):
    source = output(tmp_path)
    link = tmp_path / "link.npz"
    link.symlink_to(source)
    with ArtifactCache(tmp_path / "cache").entry(identity(), ["walk"]) as entry:
        entry.start_clip("walk", "local:1")
        with pytest.raises(CacheError):
            entry.settle_clip(
                "walk", "local:1", status="succeeded", artifacts={"clip": link}
            )


class LocalBundleReader:
    """Local byte source only; not a simulated Catalog admission."""

    def __init__(self, files):
        self.files = {sha: path.read_bytes() for sha, path in files.items()}

    def download_bundle_member(self, pack, sha, target):
        target.write_bytes(self.files[sha])
        return target


def test_restore_validates_identity_and_bytes_without_source_provider(tmp_path):
    with ArtifactCache(tmp_path / "original").entry(identity(), ["walk"]) as entry:
        manifest_sha = finish(entry, output(tmp_path))
        reader = LocalBundleReader(entry.files())
    restored = ArtifactCache(tmp_path / "restored")
    publisher = CatalogPublisher(reader, PROJECT, PROVENANCE)
    assert publisher.restore(restored, "retarget.exact", manifest_sha) == identity()
    with restored.entry(identity(), ["walk"]) as entry:
        assert entry.document["complete"]
        assert entry.complete() == manifest_sha


def test_corrupt_restore_keeps_partial_cache_and_never_completes(tmp_path):
    with ArtifactCache(tmp_path / "original").entry(identity(), ["walk"]) as entry:
        manifest_sha = finish(entry, output(tmp_path))
        reader = LocalBundleReader(entry.files())
    blob_sha = next(sha for sha in reader.files if sha != manifest_sha)
    reader.files[blob_sha] = b"bad object"
    cache = ArtifactCache(tmp_path / "restore")
    with pytest.raises(CacheError, match="disagree"):
        CatalogPublisher(reader, PROJECT, PROVENANCE).restore(
            cache, "pack", manifest_sha
        )
    assert list(cache.root.glob("restore-*"))
    with cache.entry(identity(), ["walk"]) as entry:
        assert not entry.document["complete"]
        assert entry.document["clips"]["walk"]["status"] == "pending"


def test_lost_registration_ack_refuses_new_mutation_before_sdk_import(tmp_path):
    with ArtifactCache(tmp_path / "cache").entry(identity(), ["walk"]) as entry:
        manifest_sha = finish(entry, output(tmp_path))
        document = {
            "binding": {
                "project_id": PROJECT,
                "pack": "retarget.test",
                "provenance": asdict(PROVENANCE),
                "manifest_sha256": manifest_sha,
            },
            "objects": {manifest_sha: {"registering": True}},
            "published": False,
        }
        (entry.path / "publication.json").write_text(
            json.dumps({"sha256": content_hash(document), "document": document})
        )
        with pytest.raises(AmbiguousExecution):
            CatalogPublisher(None, PROJECT, PROVENANCE).publish(entry, "retarget.test")


def test_inspection_snapshot_cannot_mutate_retained_outcomes(tmp_path):
    with ArtifactCache(tmp_path).entry(identity(), ["walk"]) as entry:
        changed = entry.document
        changed["clips"]["walk"]["status"] = "failed"
        assert entry.document["clips"]["walk"]["status"] == "pending"


def test_empty_output_publication_is_explicitly_refused_without_sdk(tmp_path):
    with ArtifactCache(tmp_path / "cache").entry(identity(), ["walk"]) as entry:
        finish(entry, output(tmp_path, b""))
        with pytest.raises(CacheError, match="nonempty"):
            CatalogPublisher(None, PROJECT, PROVENANCE).publish(entry, "retarget.test")
        assert not (entry.path / "publication.json").exists()


def test_publication_journal_corruption_refused_without_sdk(tmp_path):
    with ArtifactCache(tmp_path / "cache").entry(identity(), ["walk"]) as entry:
        finish(entry, output(tmp_path))
        state = entry.path / "publication.json"
        state.write_text(
            json.dumps({"sha256": "0" * 64, "document": {"bad": "journal"}})
        )
        with pytest.raises(CacheError, match="hash mismatch"):
            CatalogPublisher(None, PROJECT, PROVENANCE).publish(entry, "retarget.test")
        assert json.loads(state.read_text())["document"] == {"bad": "journal"}
