"""Immutable retarget outputs and restart-safe local/Catalog custody.

This module runs no solver or paid job. A running clip can only be settled under its
original execution reference; opening a cache never authorizes another attempt.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Iterator, Literal, Mapping, cast
from urllib.parse import urlsplit

from sx_contracts.identity import CanonicalDocument, canonical_json, content_digest

SCHEMA = "xmimic.retarget-cache/v1"
MAX_CATALOG_FILE_BYTES = 1024**3


class ClipDisposition(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"


TerminalDisposition = Literal["succeeded", "rejected", "failed"]
_TERMINAL = frozenset(
    {ClipDisposition.SUCCEEDED, ClipDisposition.REJECTED, ClipDisposition.FAILED}
)


class CacheError(ValueError):
    """Invalid identity, unsafe path or corrupt retained state."""


class AmbiguousExecution(CacheError):
    """Retained work must be reconciled, never silently executed again."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CacheError(message)


def _sha(value: object) -> str:
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        "Expected a lowercase SHA-256",
    )
    return str(value)


def _json(value: object) -> bytes:
    try:
        return canonical_json(cast(CanonicalDocument, value)).encode("utf-8")
    except (ValueError, TypeError) as error:
        raise CacheError("Noncanonical JSON value") from error


def content_hash(value: object) -> str:
    try:
        return str(content_digest(cast(CanonicalDocument, value)))
    except (ValueError, TypeError) as error:
        raise CacheError("Noncanonical JSON value") from error


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out = {}
    for key, value in pairs:
        _require(key not in out, "Duplicate JSON key")
        out[key] = value
    return out


def _read(path: Path) -> dict:
    _require(not path.is_symlink() and path.is_file(), "Missing or linked manifest")
    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                CacheError("Nonfinite JSON")
            ),
        )
    except (ValueError, OSError) as error:
        raise CacheError("Unreadable manifest") from error
    _require(isinstance(value, dict), "Manifest must be an object")
    return value


def _relative(value: object) -> str:
    _require(isinstance(value, str) and bool(value), "Empty logical path")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in value
        and "\x00" not in value
        and path.as_posix() == value
        and value != ".",
        "Unsafe logical path",
    )
    return value


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic(path: Path, data: bytes) -> None:
    _require(not path.is_symlink(), "Refusing linked destination")
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _file_hash(path: Path) -> tuple[str, int]:
    _require(not path.is_symlink(), "Refusing linked artifact")
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        _require(
            stat.S_ISREG(os.fstat(stream.fileno()).st_mode),
            "Artifact is not a regular file",
        )
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _verify(path: Path, sha256: str, size: int) -> None:
    _require(type(size) is int and size >= 0, "Invalid artifact size")
    _require(
        _file_hash(path) == (_sha(sha256), size),
        "Artifact bytes disagree with manifest",
    )


@dataclass(frozen=True)
class CacheIdentity:
    source_sha256: str
    shape_sha256: str
    embodiment_id: str
    urdf_sha256: str
    mapper_sha256: str
    solver_sha256: str
    settings_sha256: str

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            _sha(value)

    @property
    def key(self) -> str:
        return content_hash({"schema": SCHEMA, "identity": asdict(self)})


def _validate(document: dict) -> CacheIdentity:
    _require(
        set(document) == {"schema", "identity", "key", "clips", "complete"},
        "Unknown cache manifest fields",
    )
    _require(document["schema"] == SCHEMA, "Unknown cache schema")
    try:
        identity = CacheIdentity(**document["identity"])
    except TypeError as error:
        raise CacheError("Malformed cache identity") from error
    _require(document["key"] == identity.key, "Cache identity digest changed")
    clips = document["clips"]
    _require(isinstance(clips, dict) and bool(clips), "Cache requires a clip inventory")
    for name, clip in clips.items():
        _require(
            isinstance(name, str) and bool(name) and isinstance(clip, dict),
            "Invalid clip",
        )
        _require(
            set(clip) == {"status", "execution_ref", "reason", "artifacts", "metadata"},
            "Unknown clip fields",
        )
        status = clip["status"]
        _require(status in set(ClipDisposition), "Unknown clip disposition")
        _require(isinstance(clip["metadata"], dict), "Clip metadata must be an object")
        _require(isinstance(clip["reason"], str), "Clip reason must be text")
        _require(
            clip["execution_ref"] is None
            or isinstance(clip["execution_ref"], str)
            and bool(clip["execution_ref"]),
            "Invalid execution reference",
        )
        if status == "pending":
            _require(
                clip["execution_ref"] is None
                and not clip["reason"]
                and not clip["metadata"],
                "Pending clip already carries execution facts",
            )
        if status == "running":
            _require(
                not clip["reason"] and not clip["metadata"],
                "Running clip has terminal facts",
            )
        if status != "pending":
            _require(
                bool(clip["execution_ref"]), "Disposition lacks execution reference"
            )
        if status in {"failed", "rejected"}:
            _require(bool(clip["reason"]), "Disposition lacks reason")
        _require(isinstance(clip["artifacts"], dict), "Artifacts must be an object")
        _require(
            status in {"succeeded", "rejected"} or not clip["artifacts"],
            "Failed/pending clip has outputs",
        )
        if status == "rejected" and clip["artifacts"]:
            _require(
                set(clip["artifacts"]) <= {"candidate.npz", "retarget.json", "source-motion.json", "original-retarget.json"}
                and clip["metadata"].get("training_admitted") is False,
                "Rejected artifacts must be explicitly non-training diagnostics",
            )
        if status == "succeeded":
            _require(bool(clip["artifacts"]), "Successful clip has no outputs")
        for logical_path, artifact in clip["artifacts"].items():
            _relative(logical_path)
            _require(
                isinstance(artifact, dict)
                and set(artifact) == {"sha256", "size_bytes"},
                "Invalid artifact reference",
            )
            _sha(artifact["sha256"])
            _require(
                type(artifact["size_bytes"]) is int and artifact["size_bytes"] >= 0,
                "Invalid artifact size",
            )
    _require(type(document["complete"]) is bool, "Invalid completion flag")
    if document["complete"]:
        _require(
            all(c["status"] in _TERMINAL for c in clips.values()),
            "Incomplete inventory",
        )
    _json(document)
    return identity


class ArtifactCache:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        _require(not self.root.is_symlink(), "Cache root must not be linked")

    @contextmanager
    def entry(
        self, identity: CacheIdentity, clip_ids: list[str]
    ) -> Iterator[CacheEntry]:
        _require(
            bool(clip_ids)
            and all(isinstance(c, str) and c for c in clip_ids)
            and len(set(clip_ids)) == len(clip_ids),
            "Invalid or duplicate clip inventory",
        )
        lock = self.root / f"{identity.key}.lock"
        descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            directory = self.root / identity.key
            _require(not directory.is_symlink(), "Linked cache entry")
            new = not directory.exists()
            directory.mkdir(exist_ok=True)
            entry = CacheEntry(directory, identity, clip_ids, new=new)
            try:
                yield entry
            finally:
                entry._active = False
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


class CacheEntry:
    """Use only inside ArtifactCache.entry's process lock."""

    def __init__(
        self, path: Path, identity: CacheIdentity, clip_ids: list[str], *, new: bool
    ):
        self._active = True
        self.path, self.identity = path, identity
        self.blobs = path / "blobs"
        _require(not self.blobs.is_symlink(), "Linked blob directory")
        self.blobs.mkdir(exist_ok=True)
        self.state_path = path / "state.json"
        if self.state_path.exists():
            envelope = _read(self.state_path)
            _require(set(envelope) == {"sha256", "document"}, "Invalid state envelope")
            self._document = envelope["document"]
            _require(
                content_hash(self._document) == envelope["sha256"],
                "State hash mismatch",
            )
            _require(_validate(self._document) == identity, "Changed cache identity")
            _require(
                set(self._document["clips"]) == set(clip_ids), "Changed clip inventory"
            )
            self.verify()
        else:
            _require(new, "Existing cache lost its state; refusing to recreate work")
            self._document = {
                "schema": SCHEMA,
                "identity": asdict(identity),
                "key": identity.key,
                "complete": False,
                "clips": {
                    name: {
                        "status": "pending",
                        "execution_ref": None,
                        "reason": "",
                        "artifacts": {},
                        "metadata": {},
                    }
                    for name in clip_ids
                },
            }
            self._save()

    @property
    def document(self) -> dict:
        """A detached inspection snapshot; changes cannot rewrite retained outcomes."""
        return deepcopy(self._document)

    def _save(self) -> None:
        _require(self._active, "Cache entry lock is no longer held")
        _validate(self._document)
        _atomic(
            self.state_path,
            _json({"sha256": content_hash(self._document), "document": self._document}),
        )

    def verify(self) -> None:
        _require(self._active, "Cache entry lock is no longer held")
        _validate(self._document)
        for clip in self._document["clips"].values():
            for artifact in clip["artifacts"].values():
                _verify(
                    self.blobs / artifact["sha256"],
                    artifact["sha256"],
                    artifact["size_bytes"],
                )
        if self._document["complete"]:
            _require(
                not (self.path / "manifest.json").is_symlink(),
                "Linked completed manifest",
            )
            _require(
                (self.path / "manifest.json").read_bytes() == _json(self._document),
                "Completed manifest changed",
            )

    def start_clip(self, clip_id: str, execution_ref: str) -> None:
        _require(self._active, "Cache entry lock is no longer held")
        _require(not self._document["complete"], "Completed cache is immutable")
        _require(
            isinstance(execution_ref, str) and bool(execution_ref),
            "Execution reference required",
        )
        clip = self._document["clips"][clip_id]
        if clip["status"] != "pending":
            raise AmbiguousExecution(
                "Clip already has retained work; reconcile its execution reference"
            )
        clip.update(status="running", execution_ref=execution_ref)
        self._save()

    def frame_progress(self, clip_id: str, execution_ref: str, *, total_frames: int,
                       joint_count: int, landmark_count: int, direction_count: int = 0):
        """Start or resume only the matching, durable frame-journal execution."""
        from .retarget_progress import FrameJournal

        def assert_active():
            _require(self._active, "Cache entry lock is no longer held")
        assert_active()
        _require(not self._document["complete"], "Completed cache is immutable")
        _require(isinstance(execution_ref, str) and bool(execution_ref), "Execution reference required")
        clip = self._document["clips"][clip_id]
        pending = clip["status"] == "pending"
        _require(pending or (clip["status"] == "running" and clip["execution_ref"] == execution_ref),
                 "Progress must resume the original running execution")
        journal = FrameJournal(
            self.path / ("progress-" + content_hash({"clip_id": clip_id})),
            binding={"cache_key": self.identity.key, "clip_id": clip_id, "execution_ref": execution_ref},
            total_frames=total_frames, joint_count=joint_count, landmark_count=landmark_count,
            direction_count=direction_count,
            create=pending, assert_active=assert_active,
        )
        if pending:
            self.start_clip(clip_id, execution_ref)
        return journal

    def settle_clip(
        self,
        clip_id: str,
        execution_ref: str,
        *,
        status: TerminalDisposition,
        artifacts: Mapping[str, Path] | None = None,
        reason: str = "",
        metadata: dict | None = None,
    ) -> None:
        _require(self._active, "Cache entry lock is no longer held")
        _require(not self._document["complete"], "Completed cache is immutable")
        clip = self._document["clips"][clip_id]
        _require(
            clip["status"] == "running" and clip["execution_ref"] == execution_ref,
            "Settlement must reconcile the original running execution",
        )
        _require(status in _TERMINAL, "Expected a terminal clip disposition")
        _require(
            isinstance(reason, str) and (status == "succeeded" or bool(reason)),
            "Failed/rejected clips require a reason",
        )
        _require(
            bool(artifacts)
            if status == "succeeded"
            else (status == "rejected" or not artifacts),
            "Failed clips cannot carry output files",
        )
        if status == "rejected" and artifacts:
            _require(
                set(artifacts) <= {"candidate.npz", "retarget.json", "source-motion.json", "original-retarget.json"}
                and metadata is not None
                and metadata.get("training_admitted") is False,
                "Rejected artifacts must be explicitly non-training diagnostics",
            )
        _json(metadata if metadata is not None else {})
        references = {}
        for name, original in (artifacts or {}).items():
            _relative(name)
            _require(
                not original.is_symlink() and original.is_file(), "Unsafe output file"
            )
            descriptor, temporary = tempfile.mkstemp(prefix=".copy-", dir=self.blobs)
            try:
                with (
                    os.fdopen(descriptor, "wb") as output,
                    os.fdopen(
                        os.open(original, os.O_RDONLY | os.O_NOFOLLOW), "rb"
                    ) as source,
                ):
                    _require(
                        stat.S_ISREG(os.fstat(source.fileno()).st_mode),
                        "Not a regular output",
                    )
                    shutil.copyfileobj(source, output, 1024**2)
                    output.flush()
                    os.fsync(output.fileno())
                copied = Path(temporary)
                sha, size = _file_hash(copied)
                destination = self.blobs / sha
                if destination.exists():
                    _verify(destination, sha, size)
                else:
                    os.replace(copied, destination)
                    _fsync_dir(self.blobs)
                references[name] = {"sha256": sha, "size_bytes": size}
            finally:
                Path(temporary).unlink(missing_ok=True)
        clip.update(
            status=status,
            reason=reason,
            artifacts=references,
            metadata=deepcopy(metadata) if metadata is not None else {},
        )
        self._save()

    def complete(self) -> str:
        _require(
            all(c["status"] in _TERMINAL for c in self._document["clips"].values()),
            "Every clip needs a retained terminal disposition",
        )
        self.verify()
        self._document["complete"] = True
        manifest = _json(self._document)
        target = self.path / "manifest.json"
        if target.exists():
            _require(target.read_bytes() == manifest, "Immutable manifest conflict")
        else:
            _atomic(target, manifest)
        self._save()
        return hashlib.sha256(manifest).hexdigest()

    def files(self) -> dict[str, Path]:
        _require(self._document["complete"], "Publish only completed inventories")
        self.verify()
        result = {
            _file_hash(self.path / "manifest.json")[0]: self.path / "manifest.json"
        }
        for clip in self._document["clips"].values():
            for artifact in clip["artifacts"].values():
                result[artifact["sha256"]] = self.blobs / artifact["sha256"]
        return result


@dataclass(frozen=True)
class Provenance:
    repository: str
    revision: str
    license_id: str
    generator: str

    def __post_init__(self) -> None:
        _require(
            all(isinstance(v, str) and v for v in asdict(self).values()),
            "Retarget provenance must be explicit",
        )


class CatalogPublisher:
    """Released SDK transport. Caller supplies authenticated Project client and provenance.

    No mutation retries. Accepted operation references are read again on explicit resume.
    A lost registration acknowledgement refuses resubmission until reconciled externally.
    """

    def __init__(self, catalog, project_id: str, provenance: Provenance):
        from uuid import UUID

        self.catalog, self.project_id, self.provenance = (
            catalog,
            str(UUID(project_id)),
            provenance,
        )

    def _asset(self, sha: str, size: int):
        from sx_contracts.assets import (
            AssetFormat,
            AssetRole,
            AssetRef,
            AssetProvenance,
            ProvenancedAsset,
        )
        from sx_contracts.content import ContentBlob, Sha256Digest

        return ProvenancedAsset(
            AssetRef(
                f"urn:xmimic:retarget:sha256:{sha}",
                ContentBlob(Sha256Digest(sha), size),
                AssetFormat.OTHER,
                AssetRole.OTHER,
                "application/octet-stream",
                PurePosixPath(f"objects/{sha}"),
            ),
            AssetProvenance(
                self.provenance.repository,
                self.provenance.revision,
                f"objects/{sha}",
                self.provenance.license_id,
                self.provenance.generator,
            ),
        )

    def publish(self, entry: CacheEntry, pack: str) -> dict:
        return self.publish_files(
            entry.files(), entry.path / "manifest.json", entry.path, pack
        )

    def publish_files(
        self, files: dict[str, Path], manifest: Path, journal_directory: Path, pack: str
    ) -> dict:
        """Publish verified content-addressed files using the same durable transport.

        The caller supplies a named artifact manifest as one of these files. This
        retains the cache publisher's interrupted-upload and ambiguous-ACK rules.
        """
        _require(not journal_directory.is_symlink(), "Linked publication directory")
        journal_directory.mkdir(parents=True, exist_ok=True)
        lock = journal_directory / "publication.lock"
        _require(not lock.is_symlink(), "Linked publication lock")
        with lock.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                return self._publish_files(files, manifest, journal_directory, pack)
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _publish_files(self, files, manifest, journal_directory, pack):
        manifest_sha = _file_hash(manifest)[0]
        _require(manifest_sha in files, "Manifest must be a published member")
        _require(not journal_directory.is_symlink(), "Linked publication directory")
        journal_directory.mkdir(parents=True, exist_ok=True)
        for sha, path in files.items():
            _require(
                _sha(sha) == _file_hash(path)[0], "Publication file identity changed"
            )
        for path in files.values():
            size = path.stat().st_size
            _require(
                0 < size <= MAX_CATALOG_FILE_BYTES,
                "Catalog output files must be nonempty and at most 1 GiB; shard outputs first",
            )
        journal_path = journal_directory / "publication.json"
        binding = {
            "project_id": self.project_id,
            "pack": pack,
            "provenance": asdict(self.provenance),
            "manifest_sha256": manifest_sha,
        }
        if journal_path.exists():
            envelope = _read(journal_path)
            _require(
                set(envelope) == {"sha256", "document"}, "Invalid publication envelope"
            )
            journal = envelope["document"]
            _require(
                content_hash(journal) == envelope["sha256"], "Publication hash mismatch"
            )
            _require(journal.get("binding") == binding, "Publication binding changed")
            _require(
                set(journal) == {"binding", "objects", "published"},
                "Unknown publication fields",
            )
            _require(
                set(journal["objects"]) <= files.keys(), "Unknown published object"
            )
        else:
            journal = {"binding": binding, "objects": {}, "published": False}

        def save():
            _atomic(
                journal_path,
                _json({"sha256": content_hash(journal), "document": journal}),
            )

        save()
        if journal["published"]:
            return binding
        for state in journal["objects"].values():
            if state.get("registering") and "operation" not in state:
                raise AmbiguousExecution(
                    "Registration acknowledgement lost; reconcile its exact operation"
                )
        import httpx
        from sx_catalog_client import AssetUpload
        from sx_catalog_client.client import (
            CatalogOperation,
            CompletedStagingUpload,
            AbortedStagingUpload,
            MultipartPart,
        )
        from sx_contracts.operations import OperationRef

        declarations = []
        for sha, path in sorted(files.items()):
            size = path.stat().st_size
            declaration = self._asset(sha, size)
            declarations.append(declaration)
            state = journal["objects"].setdefault(sha, {})
            if state.get("registered"):
                continue
            if state.get("registering") and "operation" not in state:
                raise AmbiguousExecution(
                    "Registration acknowledgement lost; attach its exact operation before resume"
                )
            if "operation" not in state:
                upload = self.catalog.begin_staging_upload(
                    f"xmimic-retarget-{sha}", f"{sha}.artifact"
                )
                _require(
                    not isinstance(upload, AbortedStagingUpload),
                    "Upload aborted; reconcile before retry",
                )
                uri = urlsplit(upload.source_uri)
                _require(
                    uri.scheme in {"s3", "r2"} and not uri.query and not uri.fragment,
                    "Invalid Catalog staging reference",
                )
                facts = {
                    "upload_id": str(upload.upload_id),
                    "source_uri": upload.source_uri,
                    "part_size_bytes": upload.part_size_bytes,
                }
                _require(
                    all(k not in state or state[k] == v for k, v in facts.items()),
                    "Upload identity changed",
                )
                state.update(facts)
                parts = state.setdefault("parts", {})
                save()
                if isinstance(upload, CompletedStagingUpload):
                    _require(
                        upload.size_bytes == size, "Completed staging size mismatch"
                    )
                else:
                    _require(
                        5 * 1024**2 <= upload.part_size_bytes <= 128 * 1024**2,
                        "Unsupported multipart buffer bound",
                    )
                    with path.open("rb") as stream, httpx.Client(timeout=300) as http:
                        for number, offset in enumerate(
                            range(0, size, upload.part_size_bytes), 1
                        ):
                            if str(number) in parts:
                                continue
                            stream.seek(offset)
                            grant = self.catalog.staging_upload_part(
                                upload.upload_id, number
                            )
                            try:
                                response = http.put(
                                    grant.url,
                                    content=stream.read(upload.part_size_bytes),
                                )
                            except httpx.HTTPError:
                                raise CacheError(
                                    "Multipart transport failed; resume retained upload"
                                ) from None
                            _require(
                                200 <= response.status_code < 300,
                                f"Part upload HTTP {response.status_code}",
                            )
                            etag = response.headers.get("etag")
                            _require(bool(etag), "Multipart response lacks ETag")
                            parts[str(number)] = etag
                            save()
                    self.catalog.complete_staging_upload(
                        upload.upload_id,
                        size_bytes=size,
                        parts=tuple(
                            MultipartPart(int(n), tag)
                            for n, tag in sorted(parts.items(), key=lambda p: int(p[0]))
                        ),
                    )
                state["registering"] = True
                save()
                operation = self.catalog.register(
                    AssetUpload(state["source_uri"], declaration)
                )
                state["operation"] = {
                    "ref": str(operation.ref),
                    "resource_ref": operation.resource_ref,
                }
                save()
            operation = CatalogOperation(
                OperationRef.parse(state["operation"]["ref"]),
                state["operation"]["resource_ref"],
                AssetUpload(state["source_uri"], declaration),
            )
            _require(
                str(self.catalog.result(operation)) == sha,
                "Registered object digest changed",
            )
            state["registered"] = True
            save()
        self.catalog.register_asset_pack(pack, tuple(declarations))
        journal["published"] = True
        save()
        return binding

    def reconcile_registration(
        self, entry: CacheEntry, sha256: str, operation_ref: str, resource_ref: str
    ) -> None:
        """Read an externally recovered exact operation before accepting its receipt.

        This sends no registration. A failed or mismatched operation leaves the
        ambiguous journal intact. Source operation lookup is the caller's task.
        """
        from sx_catalog_client import AssetUpload
        from sx_catalog_client.client import CatalogOperation
        from sx_contracts.operations import OperationRef

        files = entry.files()
        _sha(sha256)
        _require(sha256 in files, "Operation does not belong to this cache")
        path = entry.path / "publication.json"
        envelope = _read(path)
        journal = envelope["document"]
        _require(
            content_hash(journal) == envelope["sha256"], "Publication hash mismatch"
        )
        binding = journal["binding"]
        _require(
            binding["project_id"] == self.project_id
            and binding["provenance"] == asdict(self.provenance)
            and binding["manifest_sha256"]
            == _file_hash(entry.path / "manifest.json")[0],
            "Publication binding changed",
        )
        state = journal["objects"][sha256]
        _require(
            state.get("registering") and "operation" not in state,
            "Only an ambiguous registration may be reconciled",
        )
        receipt = {"ref": operation_ref, "resource_ref": resource_ref}
        operation = CatalogOperation(
            OperationRef.parse(operation_ref),
            resource_ref,
            AssetUpload(
                state["source_uri"], self._asset(sha256, files[sha256].stat().st_size)
            ),
        )
        _require(
            str(self.catalog.result(operation)) == sha256,
            "Reconciled operation digest mismatch",
        )
        state.update(operation=receipt, registered=True)
        _atomic(path, _json({"sha256": content_hash(journal), "document": journal}))

    def restore(
        self, cache: ArtifactCache, pack: str, manifest_sha256: str
    ) -> CacheIdentity:
        _sha(manifest_sha256)
        # Downloads stay under the cache; failures preserve them for inspection.
        staging = Path(
            tempfile.mkdtemp(prefix=f"restore-{manifest_sha256}-", dir=cache.root)
        )
        _require(not staging.is_symlink(), "Linked restore directory")
        staging.mkdir(exist_ok=True)
        manifest = staging / "manifest.json"
        self.catalog.download_bundle_member(pack, manifest_sha256, manifest)
        _require(
            _file_hash(manifest)[0] == manifest_sha256,
            "Downloaded manifest digest mismatch",
        )
        document = _read(manifest)
        identity = _validate(document)
        _require(document["complete"], "Remote cache inventory is incomplete")
        with cache.entry(identity, list(document["clips"])) as entry:
            if entry.document["complete"]:
                _require(entry.document == document, "Existing completed cache differs")
                return identity
            _require(
                all(c["status"] == "pending" for c in entry.document["clips"].values()),
                "Restore cannot replace retained local work",
            )
            for clip in document["clips"].values():
                for artifact in clip["artifacts"].values():
                    sha, size = artifact["sha256"], artifact["size_bytes"]
                    target = entry.blobs / sha
                    if target.exists():
                        _verify(target, sha, size)
                    else:
                        temporary = staging / sha
                        self.catalog.download_bundle_member(pack, sha, temporary)
                        _verify(temporary, sha, size)
                        os.replace(temporary, target)
                        _fsync_dir(entry.blobs)
            _atomic(entry.path / "manifest.json", _json(document))
            entry._document = document
            entry._save()
            entry.verify()
        return identity
