"""A project-owned CPU→GPU workflow using the managed SentientX Job API.

Local journals remember operation identities; the platform remains the scheduler,
authority, billing and checkpoint owner. Reinvoking attaches to the same paid jobs.
"""

from __future__ import annotations

import fcntl
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
import re

from sx_contracts import ContentRef, Sha256Digest
from sx_contracts.identity import file_digest
from sx_training.artifacts import (
    ArtifactFile,
    ArtifactManifest,
    ArtifactPurpose,
    artifact_manifest_from_bytes,
    verified_artifacts,
)

from .artifact_cache import (
    CatalogPublisher,
    Provenance,
    _atomic,
    _json,
    _read,
    content_hash,
)


class Stage(StrEnum):
    RETARGET = "retarget"
    TRAIN = "train"


@dataclass(frozen=True)
class Pack:
    entry: str
    sha256: str
    manifest: ArtifactManifest
    directory: Path


@dataclass(frozen=True)
class JobSettings:
    image: str
    timeout_s: int
    spending_limit_usd_micros: int
    cpus: int
    memory_gib: int
    steps: int
    config: dict

    def __post_init__(self):
        if not re.fullmatch(r"[^@ ]+@sha256:[0-9a-f]{64}", self.image):
            raise ValueError("Use an immutable built image digest")
        if any(
            type(v) is not int or v <= 0
            for v in (
                self.timeout_s,
                self.spending_limit_usd_micros,
                self.cpus,
                self.memory_gib,
                self.steps,
            )
        ):
            raise ValueError(
                "Job resources, budget, timeout and steps must be positive integers"
            )
        if self.cpus > 128 or self.memory_gib > 1024:
            raise ValueError("Job allocation exceeds the managed contract")


def publish_inputs(
    catalog,
    *,
    project: str,
    directory: Path,
    journal: Path,
    provenance: Provenance,
    parent_entry: str,
    manifest: ArtifactManifest | None = None,
    training_eligible: bool = False,
) -> Pack:
    """Publish a closed named input pack and its source ancestry through Catalog."""
    from sentientx.sx.experience.catalog.v1 import library_pb2, sources_pb2

    if not parent_entry:
        raise ValueError("A governed source entry is required for artifact lineage")
    directory = directory.resolve(strict=True)
    if manifest is None:
        paths = sorted(directory.rglob("*"))
        if any(path.is_symlink() for path in paths):
            raise ValueError("Input closure cannot contain symlinks")
        manifest = ArtifactManifest(
            ArtifactPurpose.TRANSFORM,
            tuple(
                ArtifactFile(
                    path.relative_to(directory).as_posix(),
                    file_digest(path),
                    path.stat().st_size,
                )
                for path in paths
                if path.is_file()
            ),
            training_eligible,
        )
    named = verified_artifacts(manifest, directory)
    payload = manifest.canonical_bytes()
    digest = content_hash(manifest.to_dict())
    pack_key = content_hash({"manifest": digest, "parent": parent_entry})
    entry = "xmimic.inputs." + pack_key
    journal = journal / pack_key
    journal.mkdir(parents=True, exist_ok=True)
    manifest_path = journal / "artifact-manifest.json"
    _atomic(manifest_path, payload)
    files = {str(item.sha256): named[item.path] for item in manifest.files}
    files[digest] = manifest_path
    publisher = CatalogPublisher(catalog, project, provenance)
    publisher.publish_files(files, manifest_path, journal, entry)
    # Catalog's existing source-ancestry service owns held-out quarantine. This
    # deterministic replay is safe even when pack publication already completed.
    catalog.record_source_lineage(
        sources_pb2.RecordLineageRequest(
            entry=entry,
            parents=[
                sources_pb2.LineageParent(
                    entry=parent_entry,
                    supplied=[library_pb2.LINEAGE_SUPPLIED_REFERENCE],
                )
            ],
        )
    )
    return Pack(entry, digest, manifest, directory)


def restore_pack(catalog, entry: str, sha256: str, cache: Path) -> Pack:
    Sha256Digest(sha256)
    cache.mkdir(parents=True, exist_ok=True)
    lock = cache / (sha256 + ".lock")
    if lock.is_symlink():
        raise ValueError("Linked cache lock")
    with lock.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            return _restore_pack(catalog, entry, sha256, cache)
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _restore_pack(catalog, entry: str, sha256: str, cache: Path) -> Pack:
    """Download each content identity once, checking every byte before use."""
    Sha256Digest(sha256)
    root = cache / sha256
    if root.is_symlink():
        raise ValueError("Cache root cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "artifact-manifest.json"
    if not manifest_path.exists():
        temporary = root / "manifest.partial"
        catalog.download_bundle_member(entry, sha256, temporary)
        if file_digest(temporary) != sha256:
            raise ValueError("Downloaded artifact manifest digest differs")
        temporary.replace(manifest_path)
    if file_digest(manifest_path) != sha256:
        raise ValueError("Retained artifact manifest is corrupt")
    manifest = artifact_manifest_from_bytes(manifest_path.read_bytes())
    data = root / "files"
    data.mkdir(exist_ok=True)
    for item in manifest.files:
        target = data / item.path
        if any(part.is_symlink() for part in (target, *target.parents)):
            raise ValueError("Artifact cache contains a symlink")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = root / (str(item.sha256) + ".partial")
            catalog.download_bundle_member(entry, str(item.sha256), temporary)
            if (
                temporary.stat().st_size != item.size_bytes
                or file_digest(temporary) != item.sha256
            ):
                raise ValueError("Downloaded artifact differs from admitted identity")
            temporary.replace(target)
    verified_artifacts(manifest, data)
    return Pack(entry, sha256, manifest, data)


def training_candidate(output: Pack) -> ArtifactManifest:
    """Select only the worker's verified successful-motion closure for admission."""
    if not output.manifest.training_eligible:
        raise ValueError("Retarget source was not eligible for training")
    path = output.directory / "training-candidate.json"
    if "training-candidate.json" not in {item.path for item in output.manifest.files}:
        raise ValueError("Retarget output has no retained training candidate")
    verified_artifacts(output.manifest, output.directory)
    candidate = artifact_manifest_from_bytes(path.read_bytes())
    if (
        candidate.purpose is not ArtifactPurpose.TRANSFORM
        or not candidate.training_eligible
    ):
        raise ValueError("Retarget candidate is not an eligible transform output")
    owned = {item.path: item for item in output.manifest.files}
    if any(owned.get(item.path) != item for item in candidate.files):
        raise ValueError("Training candidate differs from the published output closure")
    verified_artifacts(candidate, output.directory)
    # This marks requested intent. The managed owner still must admit Catalog
    # source ancestry before it can schedule GPU learning.
    return ArtifactManifest(ArtifactPurpose.TRAINING, candidate.files, True)


async def submit_stage(
    client, pack: Pack, stage: Stage, settings: JobSettings, journal: Path
):
    from sentientx.sx.autonomy.train.v1 import training_pb2 as pb
    from google.protobuf.json_format import MessageToDict

    if (stage is Stage.TRAIN) != (pack.manifest.purpose is ArtifactPurpose.TRAINING):
        raise ValueError("Stage and admitted input purpose differ")
    request = pb.CreateCustomTrainingJobRequest(
        artifact_inputs=pb.ArtifactPackInputs(
            entry=pack.entry, manifest_sha256=pack.sha256
        ),
        authored_trainer=pb.AuthoredTrainer(
            image_uri=settings.image,
            entrypoint="xmimic.managed_worker:" + stage.value,
            config_json=_json(settings.config).decode(),
            steps=settings.steps,
        ),
        execution=pb.TrainingExecution(
            architecture=pb.EXECUTION_ARCHITECTURE_AMD64,
            accelerator=pb.ACCELERATOR_H100
            if stage is Stage.TRAIN
            else pb.ACCELERATOR_NONE,
            accelerator_count=1 if stage is Stage.TRAIN else 0,
            minimum_vram_mib=80000 if stage is Stage.TRAIN else 0,
            runtime_abi="cuda12" if stage is Stage.TRAIN else "python3.12",
            cpus=settings.cpus,
            memory_gib=settings.memory_gib,
            batch_size=1,
            seed=0,
        ),
        timeout=timedelta(seconds=settings.timeout_s),
        spending_limit_usd_micros=settings.spending_limit_usd_micros,
        name="XMimic " + stage.value,
    )
    key = content_hash(MessageToDict(request))
    path = journal / (stage.value + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        state = _read(path)
        if state.get("request_sha256") != key:
            raise ValueError(
                "Existing paid stage binds different inputs or settings; use another run directory"
            )
        if "operation" in state:
            return client.operation(state["operation"])
    else:
        _atomic(path, _json({"request_sha256": key}))
    operation = await client.train_custom(
        artifact_inputs=request.artifact_inputs,
        authored_trainer=request.authored_trainer,
        execution=request.execution,
        timeout=timedelta(seconds=settings.timeout_s),
        spending_limit_usd_micros=settings.spending_limit_usd_micros,
        name=request.name,
        idempotency_key="xmimic-" + key,
    )
    _atomic(path, _json({"request_sha256": key, "operation": str(operation.ref)}))
    return operation


async def completed_output(operation, catalog, cache: Path) -> Pack:
    result = await operation
    if not isinstance(result, ContentRef):
        raise ValueError("Managed XMimic job did not return an artifact receipt")
    return restore_pack(
        catalog, "training.artifacts." + str(operation.id), str(result.sha256), cache
    )
