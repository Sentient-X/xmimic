"""Pin and verify unmodified native controls; never treats acquisition as evaluation.

python tools/benchmark_native.py --root data/native-baselines
This preserves released policies/tasks as bytes without deserializing pickle checkpoints.
Native execution is exclusively an independently admitted SentientX Job.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request

from sx_contracts.identity import canonical_json


class NativeMethod(StrEnum):
    XMIMIC = "vanilla_xmimic"
    POLICYIK = "vanilla_policyik"


@dataclass(frozen=True)
class NativeSource:
    method: NativeMethod
    repository: str
    revision: str
    archive_sha256: str
    directory: str

    @property
    def url(self):
        return f"https://codeload.github.com/{self.repository}/tar.gz/{self.revision}"


SOURCES = (
    NativeSource(
        NativeMethod.XMIMIC,
        "nico-bohlinger/xmimic",
        "8f7693f96ca02c49f11dcc98cd14cf03f45df2fa",
        "d583da3e62edbca3bf9ca34bb6189dbd3b649e1a01cfe80054f9f5c3403c1113",
        "xmimic-8f7693f96ca02c49f11dcc98cd14cf03f45df2fa",
    ),
    NativeSource(
        NativeMethod.POLICYIK,
        "Tinaliu0123/PolicyIK",
        "22f3a5cddc674e88995cae18b93cd5a5cb3ddb86",
        "a53ecb515ae3c3f477c888eb0384832cb04ec20d0e80f083c2f0b4f5e0b167cf",
        "PolicyIK-22f3a5cddc674e88995cae18b93cd5a5cb3ddb86",
    ),
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archived_files(archive, prefix):
    """Parse only bounded regular source members, refusing traversal and links."""
    files = {}
    total = 0
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] != prefix
                or "\\" in member.name
            ):
                raise ValueError("Unsafe native archive path")
            if member.isdir():
                continue
            if not member.isfile() or len(path.parts) < 2:
                raise ValueError("Native archive must contain only regular files")
            name = PurePosixPath(*path.parts[1:]).as_posix()
            total += member.size
            if name in files or len(files) >= 10000 or total > 512 * 1024**2:
                raise ValueError("Duplicate or excessive native source archive")
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError("Unreadable native archive member")
            files[name] = stream.read()
    return files


def ensure_source(root, source):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{source.directory.lower()}.tar.gz"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        if partial.exists():
            raise ValueError(
                "Unresolved acquisition exists; reconcile the retained partial"
            )
        with (
            urllib.request.urlopen(source.url, timeout=30) as response,
            partial.open("xb") as out,
        ):
            size = 0
            while chunk := response.read(1024**2):
                size += len(chunk)
                if size > 128 * 1024**2:
                    raise ValueError("Native archive exceeds declared download bound")
                out.write(chunk)
        if digest(partial) != source.archive_sha256:
            raise ValueError("Native archive checksum differs; partial preserved")
        partial.rename(archive)
    if archive.is_symlink() or digest(archive) != source.archive_sha256:
        raise ValueError("Native archive checksum differs")
    expected = archived_files(archive, source.directory)
    target = root / source.directory
    target.mkdir(exist_ok=True)
    if target.is_symlink():
        raise ValueError("Native source cannot be a symlink")
    for name, payload in expected.items():
        path = target / name
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError("Native source contains a link")
        if path.exists():
            if not path.is_file() or path.read_bytes() != payload:
                raise ValueError(f"Native source modified: {name}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(payload)
    actual = {
        p.relative_to(target).as_posix()
        for p in target.rglob("*")
        if p.is_file() or p.is_symlink()
    }
    if actual != expected.keys():
        raise ValueError("Unrecorded files in immutable native source")
    return {
        "method": source.method.value,
        "repository": source.repository,
        "revision": source.revision,
        "archive_sha256": source.archive_sha256,
        "directory": source.directory,
        "files": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(expected.items())
        },
    }


def prepare(root):
    sources = [ensure_source(root, source) for source in SOURCES]
    receipt = {
        "schema": "xmimic.native-baseline-acquisition/v1",
        "status": "acquired_not_evaluated",
        "sources": sources,
        "native_commands": {
            "vanilla_xmimic": {
                "cwd": SOURCES[0].directory,
                "train": ["python", "train.py", "--log_dir", "runs/1gpu"],
                "evaluate": [
                    "python",
                    "-m",
                    "xmimic.evaluate",
                    "runs/1gpu",
                    "--reps",
                    "4",
                ],
                "checkpoint": None,
                "prerequisites": [
                    "checksum-verified40LAFANclips+G1meshes",
                    "LinuxCUDA12runtime",
                    "native3500iterationtraining",
                ],
            },
            "vanilla_policyik": {
                "cwd": SOURCES[1].directory + "/simulation",
                "evaluate_each_seed": [
                    [
                        "bash",
                        "scripts/evaluate.sh",
                        str(seed),
                        f"checkpoints/policyik_s{seed}.pth",
                    ]
                    for seed in (1, 2, 3)
                ],
                "protocol": "retain single-attempt separately from corrected+8transient attempts",
                "prerequisites": [
                    "IsaacLabv2.3.2",
                    "IsaacSim5.1.0",
                    "Torch2.7.0CUDA12.8",
                    "OpenArmbad82e23716e6941c2de78ccb978f57c78b37734",
                ],
            },
        },
        "execution_boundary": "SentientX immutable Job; no local or standalone Modal launcher",
        "measurements": None,
    }
    path = Path(root) / "acquisition.json"
    payload = canonical_json(receipt).encode()
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("Retained native acquisition differs")
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(payload)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare(args.root)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(args.root / "acquisition.json"),
                "sources": [
                    {k: v for k, v in s.items() if k != "files"}
                    for s in receipt["sources"]
                ],
            },
            indent=2,
        )
    )


class NativePhase(StrEnum):
    TRAIN = "train"
    EVALUATE_SINGLE = "evaluate_single"
    EVALUATE_DEPLOYMENT = "evaluate_deployment"


def native_commands(method, phase, seed, source, output, interpreter):
    """Native upstream commands; changing the task/simulator is not an option."""
    method, phase = NativeMethod(method), NativePhase(phase)
    if type(seed) is not int or seed not in (1, 2, 3):
        raise ValueError("Native benchmark seed must be1,2or3")
    source, output = Path(source), Path(output)
    if method is NativeMethod.XMIMIC:
        if phase is NativePhase.TRAIN:
            return [
                (
                    source,
                    [
                        interpreter,
                        "train.py",
                        "--seed",
                        str(seed),
                        "--log_dir",
                        str(output),
                    ],
                )
            ]
        if phase is not NativePhase.EVALUATE_SINGLE:
            raise ValueError("XMimic has no PolicyIK deployment protocol")
        return [
            (
                source,
                [
                    interpreter,
                    "-m",
                    "xmimic.evaluate",
                    str(output),
                    "--reps",
                    "4",
                    "--motion_dir",
                    str(source / "data/lafan1_g1"),
                ],
            )
        ]
    if phase is NativePhase.TRAIN:
        raise ValueError("PolicyIK native control evaluates released checkpoints only")
    cwd = source / "simulation"
    checkpoint = cwd / f"checkpoints/policyik_s{seed}.pth"
    task = cwd / "data/eval_tasks.npz"
    evaluator = [
        interpreter,
        "-m",
        "policyik_sim.evaluate",
        "--headless",
        "--checkpoint",
        str(checkpoint),
    ]
    commands = [
        (
            cwd,
            evaluator
            + ["--task_file", str(task), "--output", str(output / "single.npz")],
        )
    ]
    if phase is NativePhase.EVALUATE_DEPLOYMENT:
        corrected = output / "corrected_tasks.npz"
        commands.extend(
            [
                (
                    cwd,
                    [
                        interpreter,
                        "-m",
                        "policyik_sim.corrected_goals",
                        "--single",
                        str(output / "single.npz"),
                        "--task_file",
                        str(task),
                        "--output",
                        str(corrected),
                    ],
                ),
                (
                    cwd,
                    evaluator
                    + [
                        "--task_file",
                        str(corrected),
                        "--output",
                        str(output / "corrected.npz"),
                    ],
                ),
            ]
        )
        commands.extend(
            (
                cwd,
                evaluator
                + [
                    "--task_file",
                    str(task),
                    "--transient_seed",
                    str(k),
                    "--output",
                    str(output / f"transient{k}.npz"),
                ],
            )
            for k in range(8)
        )
    commands.append(
        (
            cwd,
            [interpreter, "-m", "policyik_sim.metrics", "--seed_dir", str(output)]
            + (["--single_only"] if phase is NativePhase.EVALUATE_SINGLE else []),
        )
    )
    return commands


def require_purpose(method, phase, manifest):
    """An evaluation declaration cannot dispatch an optimizer command."""
    from sx_training.artifacts import ArtifactPurpose

    method, phase = NativeMethod(method), NativePhase(phase)
    if phase is NativePhase.TRAIN:
        if (
            method is not NativeMethod.XMIMIC
            or manifest.purpose is not ArtifactPurpose.TRAINING
            or not manifest.training_eligible
        ):
            raise ValueError("Native optimizer requires eligible TRAINING admission")
    elif (
        manifest.purpose is not ArtifactPurpose.EVALUATION or manifest.training_eligible
    ):
        raise ValueError("Native evaluation requires EVALUATION and false eligibility")


def _regular_tree(root, *, boundary):
    """Reject aliases/special files before reading or writing retained custody."""
    import stat

    from sx_training.paths import rooted_path

    root = rooted_path(boundary, Path(root).relative_to(boundary))
    if not root.exists():
        return {}
    if not root.is_dir():
        raise ValueError("Native managed root is not a directory")
    files = {}
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError("Native managed tree contains a link or special file")
        files[path.relative_to(root).as_posix()] = digest(path)
    return files


def _pending_write(path, sha256):
    # Bind retained temporary bytes to both their destination and expected content.
    name_hash = hashlib.sha256(path.name.encode()).hexdigest()
    return path.with_name(f".native-write-{name_hash}-{sha256}")


def _reconcile_writes(root, expected, *, boundary):
    import os

    files = _regular_tree(root, boundary=boundary)
    for name, sha256 in expected.items():
        destination = root / name
        temporary = _pending_write(destination, sha256)
        relative = temporary.relative_to(root).as_posix()
        if relative not in files:
            continue
        if files[relative] != sha256:
            raise ValueError("Native pending write differs from expected bytes")
        if name in files:
            if files[name] != sha256:
                raise ValueError("Native pending destination changed")
        else:
            os.link(temporary, destination)
        temporary.unlink()
    return _regular_tree(root, boundary=boundary)


def _write_exact(path, payload, *, boundary):
    import os

    from sx_training.paths import rooted_path

    path = rooted_path(boundary, Path(path).relative_to(boundary))
    sha256 = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _pending_write(path, sha256)
    if temporary.exists() or temporary.is_symlink():
        _reconcile_writes(path.parent, {path.name: sha256}, boundary=boundary)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError("Native retained bytes changed")
        return
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _terminate_group(child):
    """Leader exit does not acknowledge termination of its remaining children."""
    import os
    import signal
    import time

    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        child.poll()
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def _publish_native(context, output):
    """Atomically expose exact completed bytes at the owner's required artifact root."""
    import os

    expected = _regular_tree(output, boundary=context.run_dir)
    artifact = context.run_dir / "artifact"
    current = _regular_tree(artifact, boundary=context.run_dir)
    # Only the owner execute() receipt is outside authored output identity.
    from sx_training import RECEIPT

    current.pop(RECEIPT, None)
    current.pop(f".{RECEIPT}.partial", None)
    if artifact.exists():
        if current != expected:
            raise ValueError("Native published output changed")
        return artifact
    staging = context.run_dir / "native-publication"
    retained = _reconcile_writes(staging, expected, boundary=context.run_dir)
    if any(expected.get(name) != sha for name, sha in retained.items()):
        raise ValueError("Native publication staging changed")
    staging.mkdir(exist_ok=True)
    for name in expected:
        _write_exact(
            staging / name, (output / name).read_bytes(), boundary=context.run_dir
        )
    if _regular_tree(staging, boundary=context.run_dir) != expected:
        raise ValueError("Native publication inventory differs")
    os.rename(staging, artifact)
    return artifact


def run(context):
    """Execute native steps once, retaining ambiguous paid work without rerunning it."""
    import os
    import subprocess
    import time

    config = context.config
    if set(config) != {"method", "phase", "seed"}:
        raise ValueError("Native worker requires exactly method, phase and seed")
    method, phase = NativeMethod(config["method"]), NativePhase(config["phase"])
    require_purpose(method, phase, context.artifact_manifest)
    selected = next(item for item in SOURCES if item.method is method)
    archive = context.artifacts["source.tar.gz"]
    if digest(archive) != selected.archive_sha256:
        raise ValueError("Native source differs from the pinned unmodified control")
    source = context.run_dir / "native-source"
    output = context.run_dir / "native-output"
    records = context.run_dir / "native-steps"
    # Check all writable destinations before any source copying or child execution.
    for directory in (
        source,
        output,
        records,
        context.run_dir / "artifact",
        context.run_dir / "native-publication",
    ):
        _regular_tree(directory, boundary=context.run_dir)
    payloads = archived_files(archive, selected.directory)
    if method is NativeMethod.XMIMIC:
        for row in payloads["tools/data_sha256.txt"].decode().splitlines():
            expected, name = row.split(maxsplit=1)
            name = name.lstrip(" *")
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or str(path) != name:
                raise ValueError("Unsafe upstream native asset path")
            item = context.artifacts["task/" + name]
            if digest(item) != expected:
                raise ValueError(
                    "Native G1/LAFAN asset differs from published baseline"
                )
            payloads[name] = item.read_bytes()
    expected_source = {
        name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()
    }
    retained_source = _reconcile_writes(
        source, expected_source, boundary=context.run_dir
    )
    if any(expected_source.get(name) != sha for name, sha in retained_source.items()):
        raise ValueError("Native source has modified or unrecorded files")
    source.mkdir(exist_ok=True)
    for name, payload in payloads.items():
        _write_exact(source / name, payload, boundary=context.run_dir)
    if _regular_tree(source, boundary=context.run_dir) != expected_source:
        raise ValueError("Native source inventory differs")
    output.mkdir(exist_ok=True)
    initial = {}
    if method is NativeMethod.XMIMIC and phase is not NativePhase.TRAIN:
        checkpoints = [
            (name, path)
            for name, path in context.artifacts.items()
            if name.startswith("checkpoints/")
        ]
        if not checkpoints:
            raise ValueError(
                "Native XMimic evaluation requires retained native checkpoints"
            )
        for name, path in checkpoints:
            filename = PurePosixPath(name).name
            if (
                name != "checkpoints/" + filename
                or not filename.startswith("ckpt_")
                or not filename.endswith(".npz")
            ):
                raise ValueError("Unexpected native checkpoint member")
            _write_exact(output / filename, path.read_bytes(), boundary=context.run_dir)
            initial[filename] = digest(path)
    interpreter = (
        "/opt/policyik/.venv/bin/python"
        if method is NativeMethod.POLICYIK
        else "/opt/sx/.venv/bin/python"
    )
    commands = native_commands(
        method, phase, config["seed"], source, output, interpreter
    )
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    if method is NativeMethod.POLICYIK:
        env.update(
            PYTHONPATH=str(source / "simulation") + ":/opt/policyik/openarm_isaac_lab",
            OMNI_KIT_ACCEPT_EULA="YES",
        )
    else:
        env["PYTHONPATH"] = str(source)
        env["JAX_COMPILATION_CACHE_DIR"] = str(context.cache_dir / "native-jax")
    records.mkdir(exist_ok=True)
    record_files = _regular_tree(records, boundary=context.run_dir)
    allowed_records = {f"{number:03d}.json" for number in range(len(commands))}
    if not record_files.keys() <= allowed_records:
        raise ValueError("Native journal contains unrecorded files")
    completed = []
    expected_output = initial
    pending_seen = False
    for number, (cwd, command) in enumerate(commands):
        receipt = records / f"{number:03d}.json"
        intent = {
            "command": command,
            "cwd": str(cwd),
            "source": selected.archive_sha256,
        }
        if not receipt.exists():
            pending_seen = True
            continue
        retained = json.loads(receipt.read_bytes())
        if pending_seen or retained["intent"] != intent:
            raise ValueError("Native command identity or order changed")
        if retained["state"] != "complete":
            raise RuntimeError(
                "Native paid command interrupted or ambiguous; no automatic rerun"
            )
        if any(
            retained["files"].get(name) != sha for name, sha in expected_output.items()
        ):
            raise ValueError("Native step rewrote previously completed output")
        expected_output = retained["files"]
        completed.append(retained)

    def provenance_bytes():
        return canonical_json(
            {
                "method": method.value,
                "phase": phase.value,
                "seed": config["seed"],
                "repository": selected.repository,
                "revision": selected.revision,
                "source_sha256": selected.archive_sha256,
                "steps": completed,
                "scope": "Native upstream task; not a matched stationary adaptation or physical qualification",
            }
        ).encode()

    if len(completed) == len(commands):
        _reconcile_writes(
            output,
            {"native-provenance.json": hashlib.sha256(provenance_bytes()).hexdigest()},
            boundary=context.run_dir,
        )
    actual = _regular_tree(output, boundary=context.run_dir)
    provenance = actual.pop("native-provenance.json", None)
    if actual != expected_output:
        raise ValueError("Native completed output changed or contains unrecorded files")
    if provenance is not None and len(completed) != len(commands):
        raise ValueError("Native provenance exists before completion")
    for number in range(len(completed), len(commands)):
        cwd, command = commands[number]
        receipt = records / f"{number:03d}.json"
        intent = {
            "command": command,
            "cwd": str(cwd),
            "source": selected.archive_sha256,
        }
        if context.stopping:
            raise RuntimeError("Native execution stopped before a new child command")
        _write_exact(
            receipt,
            canonical_json({"intent": intent, "state": "started"}).encode(),
            boundary=context.run_dir,
        )
        with (output / f"step-{number:03d}.log").open("xb") as log:
            log.write(b"Native command stdout/stderr follows\n")
            log.flush()
            child = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                while child.poll() is None:
                    if context.stopping:
                        raise RuntimeError(
                            "Native paid child interrupted; partial evidence retained"
                        )
                    time.sleep(0.2)
                if context.stopping:
                    raise RuntimeError(
                        "Native paid child interrupted; partial evidence retained"
                    )
                if child.returncode:
                    raise RuntimeError(
                        f"Native command failed with exit{child.returncode}; evidence retained"
                    )
            finally:
                # Always account for descendants even when the leader exited first.
                _terminate_group(child)
            log.flush()
            os.fsync(log.fileno())
        files = _regular_tree(output, boundary=context.run_dir)
        if any(files.get(name) != sha for name, sha in expected_output.items()):
            raise ValueError("Native step rewrote previously completed output")
        if _regular_tree(source, boundary=context.run_dir) != expected_source:
            raise ValueError("Native command changed immutable source")
        record = {"intent": intent, "state": "complete", "files": files}
        temporary = receipt.with_suffix(".complete")
        _write_exact(
            temporary, canonical_json(record).encode(), boundary=context.run_dir
        )
        os.replace(temporary, receipt)
        expected_output = files
        completed.append(record)
    _write_exact(
        output / "native-provenance.json", provenance_bytes(), boundary=context.run_dir
    )
    return _publish_native(context, output)
