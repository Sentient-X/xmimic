import hashlib
import io
import json
import tarfile
from pathlib import PurePosixPath

import pytest

from tools import prepare_native_inputs as prepare
from tools.benchmark_native import NativeMethod, NativeSource


def fixture(root, monkeypatch, method):
    root.mkdir()
    task = b"native motion"
    sha = hashlib.sha256(task).hexdigest()
    files = {
        f"data/motion-{n}.csv": {"sha256": sha, "size_bytes": len(task)}
        for n in range(75)
    }
    rows = "\n".join(f"{sha} {name}" for name in files).encode()
    archive = root / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as out:
        member = tarfile.TarInfo("fixture/tools/data_sha256.txt")
        member.size = len(rows)
        out.addfile(member, io.BytesIO(rows))
    selected = NativeSource(
        method,
        "example/native",
        "pinned",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
        "fixture",
    )
    monkeypatch.setattr(prepare, "SOURCES", (selected,))
    for name in files:
        path = root / "xmimic-task" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(task)
    (root / "xmimic-task.json").write_text(
        json.dumps({"files": files, "revision": "pinned"})
    )


def test_policyik_checkpoint_inputs_are_evaluation_only(tmp_path, monkeypatch):
    root = tmp_path / "root"
    fixture(root, monkeypatch, NativeMethod.POLICYIK)
    sha, manifest = prepare.prepare(root, NativeMethod.POLICYIK, tmp_path / "output")
    assert (
        manifest.purpose.value == "evaluation" and manifest.training_eligible is False
    )
    assert len(manifest.files) == 2
    assert prepare.prepare(root, NativeMethod.POLICYIK, tmp_path / "output")[0] == sha


def test_training_closure_uses_archived_upstream_checksums(tmp_path, monkeypatch):
    root = tmp_path / "root"
    fixture(root, monkeypatch, NativeMethod.XMIMIC)
    sha, manifest = prepare.prepare(root, NativeMethod.XMIMIC, tmp_path / "output")
    assert manifest.training_eligible and len(manifest.files) == 77
    path = root / "xmimic-task/data/motion-0.csv"
    path.write_text("changed native motion")
    facts = json.loads((root / "xmimic-task.json").read_text())
    facts["files"]["data/motion-0.csv"]["sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    (root / "xmimic-task.json").write_text(json.dumps(facts))
    with pytest.raises(ValueError, match="upstream"):
        prepare.prepare(root, NativeMethod.XMIMIC, tmp_path / "output")


def test_linked_output_never_writes_outside_custody(tmp_path, monkeypatch):
    root = tmp_path / "root"
    fixture(root, monkeypatch, NativeMethod.POLICYIK)
    output, outside = tmp_path / "output", tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "inputs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Linked"):
        prepare.prepare(root, NativeMethod.POLICYIK, output)
    assert list(outside.iterdir()) == []


def test_extra_input_does_not_enter_native_pack(tmp_path, monkeypatch):
    root = tmp_path / "root"
    fixture(root, monkeypatch, NativeMethod.POLICYIK)
    output = tmp_path / "output"
    prepare.prepare(root, NativeMethod.POLICYIK, output)
    (output / "inputs/foreign.py").write_text("not admitted")
    with pytest.raises(ValueError, match="Unrecorded"):
        prepare.prepare(root, NativeMethod.POLICYIK, output)


def composition_fixture(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from sx_contracts.assets import AssetRef, AssetRole, AssetFormat
    from sx_contracts.content import ContentBlob
    from sx_contracts import Sha256Digest
    from sx_training import SCHEMA
    from sx_training.artifacts import ArtifactFile, ArtifactManifest, ArtifactPurpose
    from xmimic.artifact_cache import CatalogPublisher, Provenance
    from sentientx.sx.autonomy.train.v1 import training_pb2

    root = tmp_path / "root"
    fixture(root, monkeypatch, NativeMethod.XMIMIC)
    source_sha, source = prepare.prepare(root, NativeMethod.XMIMIC, tmp_path / "source")

    def hashed(value):
        return hashlib.sha256(value).hexdigest()

    output_bytes = {
        f"ckpt_{n:06d}.npz": f"paid weights {n}".encode() for n in range(100, 3501, 100)
    }
    selected = prepare.SOURCES[0]
    provenance = {
        "method": "vanilla_xmimic",
        "phase": "train",
        "seed": 1,
        "repository": selected.repository,
        "revision": selected.revision,
        "source_sha256": selected.archive_sha256,
        "steps": [
            {
                "state": "complete",
                "files": {k: hashed(v) for k, v in output_bytes.items()},
            }
        ],
    }
    output_bytes["native-provenance.json"] = json.dumps(provenance).encode()
    receipt = {
        "schema": SCHEMA,
        "input_sha256": "1" * 64,
        "spec_sha256": "2" * 64,
        "files": {k: hashed(v) for k, v in output_bytes.items()},
    }
    output_bytes["training-receipt.json"] = json.dumps(receipt).encode()
    output = ArtifactManifest(
        ArtifactPurpose.TRANSFORM,
        tuple(
            ArtifactFile(name, Sha256Digest(hashed(value)), len(value))
            for name, value in output_bytes.items()
        ),
        True,
    )
    output_sha = hashed(output.canonical_bytes())
    source_entry, output_entry = "native.source", "training.artifacts.nativejob"
    content = {
        (source_entry, source_sha): source.canonical_bytes(),
        (output_entry, output_sha): output.canonical_bytes(),
    }
    content.update(
        {(output_entry, hashed(value)): value for value in output_bytes.values()}
    )
    content.update(
        {
            (source_entry, str(item.sha256)): (
                tmp_path / "source/inputs" / item.path
            ).read_bytes()
            for item in source.files
        }
    )
    refs = {
        key: AssetRef(
            (
                f"urn:xmimic:retarget:sha256:{key[1]}"
                if key[0] == source_entry
                else f"catalog://assets/{key[1]}"
            ),
            ContentBlob(Sha256Digest(key[1]), len(value)),
            AssetFormat.OTHER,
            AssetRole.OTHER,
            "application/octet-stream",
            PurePosixPath("objects/" + key[1]) if key[0] == source_entry else None,
        )
        for key, value in content.items()
    }
    packs = {
        entry: {
            "name": entry,
            "files": [
                {
                    "sha256": sha,
                    "size_bytes": ref.content.size_bytes,
                    "role": ref.role.value,
                    "path": str(ref.logical_path)
                    if ref.logical_path is not None
                    else None,
                }
                for (parent, sha), ref in refs.items()
                if parent == entry
            ],
        }
        for entry in (source_entry, output_entry)
    }
    lineage = {output_entry: {source_entry}}

    class Catalog:
        fail = None
        registrations = 0
        lineage_writes = 0
        downloads = []

        def read_asset(self, entry, sha):
            return SimpleNamespace(asset=refs[(entry, str(sha))])

        def download_bundle_member(self, entry, sha, path):
            value = content[(entry, sha)]
            assert not value.startswith(b"paid weights"), (
                "Checkpoint bytes must never cross laptop"
            )
            self.downloads.append(sha)
            path.write_bytes(value)

        def register_asset_pack(self, entry, assets):
            self.registrations += 1
            packs[entry] = {
                "name": entry,
                "files": [
                    {
                        "sha256": a.sha256,
                        "size_bytes": a.content.size_bytes,
                        "role": a.role.value,
                        "path": None if a.logical_path is None else str(a.logical_path),
                    }
                    for a in assets
                ],
            }
            refs.update({(entry, str(a.sha256)): a for a in assets})
            self.retained_assets = assets
            if self.fail == "pack":
                self.fail = None
                raise OSError("accepted pack ACK lost")

        def record_source_lineage(self, request):
            self.lineage_writes += 1
            lineage[request.entry] = {p.entry for p in request.parents}
            if self.fail == "lineage":
                self.fail = None
                raise OSError("accepted lineage ACK lost")

    catalog = Catalog()

    class Publisher(CatalogPublisher):
        promoted = []

        def publish_files(self, files, manifest, journal_directory, pack):
            assert len(files) == 1 and next(iter(files.values())) == manifest
            self.promoted.append(manifest.read_bytes())

    origin = Provenance("https://example.com/native", "original", "MIT", "native")
    publisher = Publisher(catalog, "b98ba48a-39c1-4346-98a4-d9c130144a42", origin)
    job = training_pb2.TrainingJob(
        id="nativejob", status=training_pb2.TRAINING_JOB_STATUS_SUCCEEDED
    )
    job.authored_trainer.entrypoint = "tools.benchmark_native:run"
    job.authored_trainer.config_json = (
        '{"method":"vanilla_xmimic","phase":"train","seed":1}'
    )
    job.authored_trainer.steps = 3500
    job.execution.seed = 1
    job.artifact_inputs.entry = source_entry
    job.artifact_inputs.manifest_sha256 = source_sha
    kwargs = dict(
        job=job,
        result=SimpleNamespace(sha256=output_sha),
        source_provenance=origin,
        directory=tmp_path / "compose",
        read_pack=packs.get,
        read_lineage=lambda name: lineage.get(name, set()),
    )
    return catalog, publisher, kwargs, refs, source, output, provenance, receipt


@pytest.mark.parametrize("fault", [None, "pack", "lineage"])
def test_metadata_handoff_reconciles_accepted_mutations_without_reupload(
    tmp_path, monkeypatch, fault
):
    catalog, publisher, args, refs, *_ = composition_fixture(tmp_path, monkeypatch)
    catalog.fail = fault
    if fault:
        with pytest.raises(OSError, match="ACK lost"):
            prepare.compose_evaluation(catalog, publisher, **args)
    result = prepare.compose_evaluation(catalog, publisher, **args)
    assert result["purpose"] == "evaluation" and result["training_eligible"] is False
    assert result["files"] == 114  # 77 original +35 checkpoints +2 provenance documents
    assert catalog.registrations == catalog.lineage_writes == 1
    assert len(publisher.promoted) == 1
    assert len(catalog.downloads) == 4
    actual = {a.sha256: a for a in catalog.retained_assets}
    for (entry, sha), ref in refs.items():
        if sha in actual:
            assert actual[sha] == ref
    assert prepare.compose_evaluation(catalog, publisher, **args) == result
    assert catalog.registrations == 1 and len(publisher.promoted) == 1


@pytest.mark.parametrize(
    "change", ["parent", "size", "source_sha", "receipt", "partial", "seed"]
)
def test_handoff_refuses_unproven_or_changed_paid_output(tmp_path, monkeypatch, change):
    from dataclasses import replace

    catalog, publisher, args, refs, source, output, provenance, receipt = (
        composition_fixture(tmp_path, monkeypatch)
    )
    if change == "parent":
        args["read_lineage"] = lambda _: set()
    elif change == "size":
        key = next(
            key
            for key in refs
            if key[0] == "native.source"
            and key[1] != args["job"].artifact_inputs.manifest_sha256
        )
        row = next(
            row for row in args["read_pack"](key[0])["files"] if row["sha256"] == key[1]
        )
        row["size_bytes"] += 1
    elif change == "source_sha":
        source = replace(
            source,
            files=tuple(
                replace(f, sha256="0" * 64) if f.path == "source.tar.gz" else f
                for f in source.files
            ),
        )
    elif change == "receipt":
        receipt["files"].pop("ckpt_003500.npz")
    elif change == "partial":
        output = replace(
            output, files=tuple(f for f in output.files if f.path != "ckpt_003500.npz")
        )
    else:
        provenance["seed"] = 2
    with pytest.raises(ValueError):
        if change in {"parent", "size"}:
            prepare.compose_evaluation(catalog, publisher, **args)
        else:
            prepare.evaluation_manifest(source, output, provenance, receipt, seed=1)
    assert not publisher.promoted and not catalog.registrations


def test_handoff_refuses_changed_journal_binding_and_links(tmp_path, monkeypatch):
    catalog, publisher, args, *_ = composition_fixture(tmp_path, monkeypatch)
    prepare.compose_evaluation(catalog, publisher, **args)
    args["job"].id = "different-job"
    with pytest.raises(ValueError, match="identity changed"):
        prepare.compose_evaluation(catalog, publisher, **args)


def test_unacknowledged_unobserved_registration_is_not_retried(tmp_path, monkeypatch):
    catalog, publisher, args, *_ = composition_fixture(tmp_path, monkeypatch)
    calls = []

    def interrupted(entry, assets):
        calls.append(entry)
        raise OSError("write disposition unknown")

    catalog.register_asset_pack = interrupted
    with pytest.raises(OSError, match="unknown"):
        prepare.compose_evaluation(catalog, publisher, **args)
    with pytest.raises(RuntimeError, match="Ambiguous pack"):
        prepare.compose_evaluation(catalog, publisher, **args)
    assert len(calls) == 1 and len(publisher.promoted) == 1


def test_linked_composition_directory_never_writes_outside(tmp_path, monkeypatch):
    catalog, publisher, args, *_ = composition_fixture(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    args["directory"].symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Linked composition"):
        prepare.compose_evaluation(catalog, publisher, **args)
    assert list(outside.iterdir()) == [] and not publisher.promoted


def test_composition_refuses_changed_uri_even_when_content_metadata_matches(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    catalog, publisher, args, refs, *_ = composition_fixture(tmp_path, monkeypatch)
    original = catalog.register_asset_pack

    def changed(entry, assets):
        original(entry, assets)
        key = (entry, str(assets[0].sha256))
        refs[key] = replace(refs[key], location="unrelated://foreign/reference")

    catalog.register_asset_pack = changed
    with pytest.raises(ValueError, match="AssetRef changed"):
        prepare.compose_evaluation(catalog, publisher, **args)
    assert catalog.lineage_writes == 0
