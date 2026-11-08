import hashlib
import io
import tarfile

import pytest

from tools.benchmark_native import (
    NativeMethod,
    NativeSource,
    archived_files,
    ensure_source,
)


def source(tmp_path, members):
    archive = tmp_path / "fixture-revision.tar.gz"
    with tarfile.open(archive, "w:gz") as out:
        for name, kind, payload in members:
            member = tarfile.TarInfo("fixture-revision/" + name)
            member.type = kind
            member.size = len(payload)
            member.linkname = "/outside"
            out.addfile(member, io.BytesIO(payload))
    return NativeSource(
        NativeMethod.POLICYIK,
        "example/source",
        "revision",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
        "fixture-revision",
    )


def test_exact_reproduction_refuses_modified_or_extra_source(tmp_path):
    spec = source(tmp_path, [("script.py", tarfile.REGTYPE, b"original")])
    first = ensure_source(tmp_path, spec)
    assert first == ensure_source(tmp_path, spec)
    path = tmp_path / spec.directory / "script.py"
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="modified"):
        ensure_source(tmp_path, spec)
    assert path.read_bytes() == b"modified"
    path.write_bytes(b"original")
    (path.parent / "shadow.py").write_bytes(b"shadow")
    with pytest.raises(ValueError, match="Unrecorded"):
        ensure_source(tmp_path, spec)


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escaped", tarfile.REGTYPE),
        ("link", tarfile.SYMTYPE),
        ("link", tarfile.LNKTYPE),
    ],
)
def test_archive_refuses_traversal_and_links(tmp_path, name, kind):
    spec = source(tmp_path, [(name, kind, b"bad")])
    with pytest.raises(ValueError):
        archived_files(tmp_path / "fixture-revision.tar.gz", spec.directory)
    assert not (tmp_path / "escaped").exists()


def test_checksum_drift_refuses_before_source_materialization(tmp_path):
    spec = source(tmp_path, [("policy.pth", tarfile.REGTYPE, b"not executed")])
    with (tmp_path / "fixture-revision.tar.gz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        ensure_source(tmp_path, spec)
    assert not (tmp_path / spec.directory).exists()


def test_native_commands_keep_single_and_deployment_protocols_separate(tmp_path):
    from tools.benchmark_native import NativePhase, native_commands

    single = native_commands(
        NativeMethod.POLICYIK,
        NativePhase.EVALUATE_SINGLE,
        1,
        tmp_path / "source",
        tmp_path / "output",
        "/pinned/python",
    )
    deployment = native_commands(
        NativeMethod.POLICYIK,
        NativePhase.EVALUATE_DEPLOYMENT,
        1,
        tmp_path / "source",
        tmp_path / "output",
        "/pinned/python",
    )
    assert len(single) == 2 and "--single_only" in single[-1][1]
    assert len(deployment) == 12 and "--single_only" not in deployment[-1][1]
    assert sum("--transient_seed" in command for _, command in deployment) == 8
    assert all("train.py" not in command for _, command in deployment)
    with pytest.raises(ValueError, match="released"):
        native_commands(
            NativeMethod.POLICYIK, NativePhase.TRAIN, 1, tmp_path, tmp_path, "/python"
        )


def test_native_worker_refuses_optimizer_under_evaluation_intent():
    from types import SimpleNamespace
    from sx_training.artifacts import ArtifactPurpose
    from tools.benchmark_native import NativePhase, require_purpose

    held = SimpleNamespace(purpose=ArtifactPurpose.EVALUATION, training_eligible=False)
    require_purpose(NativeMethod.POLICYIK, NativePhase.EVALUATE_SINGLE, held)
    with pytest.raises(ValueError, match="TRAINING"):
        require_purpose(NativeMethod.XMIMIC, NativePhase.TRAIN, held)
    train = SimpleNamespace(purpose=ArtifactPurpose.TRAINING, training_eligible=True)
    require_purpose(NativeMethod.XMIMIC, NativePhase.TRAIN, train)
    with pytest.raises(ValueError, match="EVALUATION"):
        require_purpose(NativeMethod.XMIMIC, NativePhase.EVALUATE_SINGLE, train)


@pytest.mark.parametrize("exit_code", [0, 9])
def test_native_paid_child_is_never_reexecuted_after_result_or_failure(
    tmp_path, monkeypatch, exit_code
):
    import sys
    from types import SimpleNamespace
    import tools.benchmark_native as native
    from sx_training.artifacts import ArtifactPurpose

    selected = source(tmp_path, [("script.py", tarfile.REGTYPE, b"original")])
    monkeypatch.setattr(native, "SOURCES", (selected,))
    counter = tmp_path / "calls"
    child = tmp_path / "child.py"
    child.write_text(
        "from pathlib import Path\nimport sys\n"
        f"p=Path({str(counter)!r}); p.write_text(p.read_text()+'x' if p.exists() else 'x')\n"
        "Path(sys.argv[1]).write_bytes(b'native result')\n"
        f"print('native stdout',flush=True)\nraise SystemExit({exit_code})\n"
    )
    monkeypatch.setattr(
        native,
        "native_commands",
        lambda method, phase, seed, source, output, interpreter: [
            (source, [sys.executable, str(child), str(output / "result.bin")])
        ],
    )
    context = SimpleNamespace(
        config={"method": "vanilla_policyik", "phase": "evaluate_single", "seed": 1},
        artifact_manifest=SimpleNamespace(
            purpose=ArtifactPurpose.EVALUATION, training_eligible=False
        ),
        artifacts={"source.tar.gz": tmp_path / "fixture-revision.tar.gz"},
        run_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
        stopping=False,
    )
    context.run_dir.mkdir()
    if exit_code:
        with pytest.raises(RuntimeError, match="failed"):
            native.run(context)
        with pytest.raises(RuntimeError, match="no automatic rerun"):
            native.run(context)
    else:
        output = native.run(context)
        assert native.run(context) == output
        (output / "result.bin").write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="output changed"):
            native.run(context)
    assert counter.read_text() == "x"
    assert (
        context.run_dir / "native-output/step-000.log"
    ).read_text() == "Native command stdout/stderr follows\nnative stdout\n"


def managed_fixture(tmp_path, monkeypatch, child, *, linked_mounts=False):
    import json
    import sys
    import tools.benchmark_native as native
    from sx_contracts import Sha256Digest
    from sx_training import TrainingContext
    from sx_training.artifacts import ArtifactFile, ArtifactManifest, ArtifactPurpose

    selected = source(tmp_path, [("script.py", tarfile.REGTYPE, child.encode())])
    if linked_mounts:
        for name in ("data", "runs", "cache"):
            physical = tmp_path / "provider-targets" / name
            physical.mkdir(parents=True)
            (tmp_path / name).symlink_to(physical, target_is_directory=True)
    data = tmp_path / "data"
    (data / "inputs").mkdir(parents=True)
    archive = data / "inputs/source.tar.gz"
    archive.write_bytes((tmp_path / "fixture-revision.tar.gz").read_bytes())
    manifest = ArtifactManifest(
        ArtifactPurpose.EVALUATION,
        (
            ArtifactFile(
                "source.tar.gz",
                Sha256Digest(selected.archive_sha256),
                archive.stat().st_size,
            ),
        ),
        False,
    )
    document = {
        "schema": "sx.trainer/v1",
        "trainer": {
            "entrypoint": "tools.benchmark_native:run",
            "config": {
                "method": "vanilla_policyik",
                "phase": "evaluate_single",
                "seed": 1,
            },
        },
        "inputs": {
            "job_id": "ftjob_" + "a" * 32,
            "spec": {
                "artifact_inputs": {
                    "manifest_json": manifest.canonical_bytes().decode()
                },
            },
        },
    }
    context = TrainingContext(
        json.dumps(document).encode(),
        data_root=data,
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
    )
    monkeypatch.setattr(native, "SOURCES", (selected,))
    monkeypatch.setattr(
        native,
        "native_commands",
        lambda method, phase, seed, source, output, interpreter: [
            (source, [sys.executable, "script.py", str(output)]),
        ],
    )
    return context


SUCCESS_CHILD = "from pathlib import Path\nimport sys\nPath(sys.argv[1],'result').write_text('success')\n"


def test_real_owner_execute_accepts_atomic_native_artifact_and_reentry(
    tmp_path, monkeypatch
):
    from sx_training import execute, verify_receipt
    from sx_contracts.identity import content_digest

    context = managed_fixture(tmp_path, monkeypatch, SUCCESS_CHILD)
    artifact = execute(context)
    assert artifact == context.run_dir / "artifact"
    assert (artifact / "result").read_text() == "success"
    verify_receipt(artifact, spec_digest=content_digest(context.inputs["spec"]))
    assert execute(context) == artifact
    verify_receipt(artifact, spec_digest=content_digest(context.inputs["spec"]))


def test_unrecorded_source_refused_before_child_import(tmp_path, monkeypatch):
    import tools.benchmark_native as native

    context = managed_fixture(tmp_path, monkeypatch, "import shadow\n" + SUCCESS_CHILD)
    directory = context.run_dir / "native-source"
    directory.mkdir()
    (directory / "shadow.py").write_text("raise RuntimeError('executed unverified')")
    with pytest.raises(ValueError, match="unrecorded"):
        native.run(context)
    assert not (context.run_dir / "native-steps/000.json").exists()


@pytest.mark.parametrize(
    "name",
    [
        "native-source",
        "native-output",
        "native-steps",
        "artifact",
        "native-publication",
    ],
)
def test_managed_symlink_destinations_refused_before_writes(
    tmp_path, monkeypatch, name
):
    import tools.benchmark_native as native

    context = managed_fixture(tmp_path, monkeypatch, SUCCESS_CHILD)
    outside = tmp_path / "outside"
    outside.mkdir()
    (context.run_dir / name).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="link"):
        native.run(context)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("kind", ["extra", "link", "changed"])
@pytest.mark.parametrize("directory", ["native-output", "artifact"])
def test_completed_inventory_cannot_gain_unverified_results(
    tmp_path, monkeypatch, kind, directory
):
    from sx_training import execute

    context = managed_fixture(tmp_path, monkeypatch, SUCCESS_CHILD)
    execute(context)
    root = context.run_dir / directory
    if kind == "link":
        (root / "extra-score.json").symlink_to(root / "result")
    elif kind == "extra":
        (root / "extra-score.json").write_text('{"score":999}')
    else:
        (root / "result").write_text("changed")
    with pytest.raises(ValueError, match="changed|unrecorded|link"):
        execute(context)


def test_stopping_kills_sigterm_ignoring_grandchild_after_leader_exits(
    tmp_path, monkeypatch
):
    import time
    import psutil
    import tools.benchmark_native as native

    child = """from pathlib import Path
import subprocess, sys, time
p=Path(sys.argv[1])
g=subprocess.Popen([sys.executable,'-c',"import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"])
(p/'grandchild.pid').write_text(str(g.pid))
time.sleep(.2)
(p/'ready').write_text('yes')
time.sleep(60)
"""
    context = managed_fixture(tmp_path, monkeypatch, child)
    context._cancelled = lambda: (context.run_dir / "native-output/ready").exists()
    with pytest.raises(RuntimeError, match="interrupted"):
        native.run(context)
    pid = int((context.run_dir / "native-output/grandchild.pid").read_text())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and psutil.pid_exists(pid):
        if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
            break  # terminated; OS init owns reaping orphaned processes
        time.sleep(0.05)
    assert (
        not psutil.pid_exists(pid)
        or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    )
    with pytest.raises(RuntimeError, match="no automatic rerun"):
        native.run(context)


def test_interrupted_artifact_copy_resumes_without_executing_paid_child(
    tmp_path, monkeypatch
):
    from sx_training import execute
    import tools.benchmark_native as native

    context = managed_fixture(tmp_path, monkeypatch, SUCCESS_CHILD)
    original = native._write_exact
    with monkeypatch.context() as interrupted:

        def fail_second_publication(path, payload, **kwargs):
            if path.parent.name == "native-publication" and path.name == "result":
                raise OSError("publication interrupted")
            return original(path, payload, **kwargs)

        interrupted.setattr(native, "_write_exact", fail_second_publication)
        with pytest.raises(OSError, match="publication interrupted"):
            execute(context)
    assert not (context.run_dir / "artifact").exists()
    journal = (context.run_dir / "native-steps/000.json").read_bytes()
    monkeypatch.setattr(
        native.subprocess
        if hasattr(native, "subprocess")
        else __import__("subprocess"),
        "Popen",
        lambda *a, **k: pytest.fail("paid work restarted"),
    )
    artifact = execute(context)
    assert (artifact / "result").read_text() == "success"
    assert (context.run_dir / "native-steps/000.json").read_bytes() == journal


@pytest.mark.parametrize("after_link", [False, True])
@pytest.mark.parametrize(
    "destination_name", ["native-publication", "native-provenance.json"]
)
def test_fsynced_publication_write_reconciles_exact_destination(
    tmp_path, monkeypatch, after_link, destination_name
):
    import os
    import subprocess
    from pathlib import Path
    from sx_training import execute

    context = managed_fixture(tmp_path, monkeypatch, SUCCESS_CHILD)
    original = os.link
    with monkeypatch.context() as interrupted:

        def stop_at_link(source, destination, **kwargs):
            if (
                Path(destination).parent.name == destination_name
                or Path(destination).name == destination_name
            ):
                if after_link:
                    original(source, destination, **kwargs)
                raise OSError("interrupted after temp fsync")
            return original(source, destination, **kwargs)

        interrupted.setattr(os, "link", stop_at_link)
        with pytest.raises(OSError, match="after temp fsync"):
            execute(context)
    assert list(context.run_dir.rglob(".native-write-*"))
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: pytest.fail("paid child rerun")
    )
    artifact = execute(context)
    assert (artifact / "result").read_text() == "success"
    assert not list(artifact.glob(".native-write-*"))


def test_unknown_temporary_name_is_not_ignored(tmp_path, monkeypatch):
    from sx_training import execute

    context = managed_fixture(tmp_path, monkeypatch, SUCCESS_CHILD)
    staging = context.run_dir / "native-publication"
    staging.mkdir()
    (staging / ".native-write-unbound").write_text("unknown")
    with pytest.raises(ValueError, match="staging changed"):
        execute(context)
    assert (staging / ".native-write-unbound").read_text() == "unknown"


def test_native_custody_survives_provider_mount_aliases_and_target_rebinding(
    tmp_path, monkeypatch
):
    from sx_training import TrainingContext, execute, execution_bytes, verify_receipt
    from sx_contracts.identity import content_digest
    import subprocess

    context = managed_fixture(tmp_path, monkeypatch, SUCCESS_CHILD, linked_mounts=True)
    artifact = execute(context)
    verify_receipt(artifact, spec_digest=content_digest(context.inputs["spec"]))
    assert context.run_dir.parent == tmp_path / "runs"
    assert context.cache_dir == tmp_path / "cache"
    journal = (context.run_dir / "native-steps/000.json").read_bytes()
    assert str(tmp_path / "runs").encode() in journal
    assert b"provider-targets" not in journal
    # A new Sandbox can bind the same durable volume at a different physical target.
    old_target = (tmp_path / "runs").resolve()
    new_target = old_target.with_name("next-generation-runs")
    old_target.rename(new_target)
    (tmp_path / "runs").unlink()
    (tmp_path / "runs").symlink_to(new_target, target_is_directory=True)
    resumed = TrainingContext(
        execution_bytes(context.inputs, context.trainer),
        data_root=context.data_root,
        cache_dir=context.cache_dir,
        runs_dir=tmp_path / "runs",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Paid native command must not rerun")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    assert execute(resumed) == artifact
    assert (resumed.run_dir / "native-steps/000.json").read_bytes() == journal
