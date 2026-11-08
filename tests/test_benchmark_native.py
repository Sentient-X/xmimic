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
    ).read_text() == "native stdout\n"
