import hashlib
import io
import json
import tarfile
from pathlib import Path

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
