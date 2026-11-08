"""Private, byte-exact BONES-SEED transport through Catalog; no training admission.

Run with the installed sentientx Python. ``mirror`` reads a pinned mirror-plan;
``restore`` needs only Catalog, pack name and the manifest digest from the receipt.
Four isolated workers mirror independent files; hashes remain sequential per file.
Interrupted commands resume from their journal. Failed operations stop for inspection.
Chunks are pruned after Catalog independently verifies their hashes; client readback
is mandatory for the manifest. Only tool-owned staging is pruned, never raw/.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import copy
import threading
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
from urllib.parse import quote, urljoin, urlparse
from uuid import UUID

REPO = "bones-studio/seed"
REVISION = "2f59b2077b9da34dd4e43618e705c7cb962c9a66"
MAX_CHUNK = 1024**3
BLOCK = 1024**2
TRANSFER_SECONDS = 1800
PROGRESS_SECONDS = 30
WORKERS = 4
_PRINT_LOCK = threading.Lock()
SCHEMA = "xmimic.bones-seed-mirror/v1"
# Canonical source inventory from the pinned HF revision: path, byte size, Git blob,
# integrity algorithm and LFS content hash. Git files bind through their blob SHA-1.
EXPECTED_INVENTORY = (541, 114370281701,
    "b1b79bbce35df8749740373218e107763e0c514b68adfd763d0d8d8d2929c220")


class MirrorError(ValueError):
    """Malformed transport facts or mismatched immutable bytes."""


class MultipartTransferError(MirrorError):
    """Only controlled HTTP metadata; never retain a response body or signed URL."""
    def __init__(self, status, part_number):
        self.http_status = status
        self.part_number = part_number
        super().__init__(f"Multipart transfer failed (HTTP {status}, part {part_number})")


def progress(event, **facts):
    with _PRINT_LOCK:
        print(json.dumps({"event": event, "utc_epoch_seconds": int(time.time()), **facts}), flush=True)


def failure_facts(error):
    facts = {"error_type": type(error).__name__}
    status = getattr(getattr(getattr(error, "__cause__", None), "response", None), "status_code", None)
    if type(status) is int and 100 <= status <= 599:
        facts["http_status"] = status
    if isinstance(error, MultipartTransferError):
        facts.update(reason="multipart_http_failure", http_status=error.http_status,
                     part_number=error.part_number)
    return facts


def require(condition, message):
    if not condition:
        raise MirrorError(message)


def digest(value, length=64):
    require(isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value), "Invalid digest")
    return value


def positive(value):
    require(type(value) is int and value > 0, "Expected positive integer")
    return value


def relative(value):
    require(isinstance(value, str) and value and "\\" not in value and "\x00" not in value, "Invalid path")
    p = PurePosixPath(value)
    require(not p.is_absolute() and ".." not in p.parts and value == p.as_posix() and p.parts, "Unsafe path")
    return value


def under(root, name):
    target = root / relative(name)
    require(target.absolute() == target.resolve(), "Symlink path refused")
    return target


def document(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def atomic(path, content):
    require(path.absolute() == path.resolve(), "Symlink checkpoint refused")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".mirror-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        d = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(d)
        finally:
            os.close(d)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def hashes(path):
    size = path.stat().st_size
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        while block := stream.read(BLOCK):
            sha.update(block)
            git.update(block)
    return size, sha.hexdigest(), git.hexdigest()


def check_bytes(path, size, sha):
    actual = hashes(path)
    require(actual[:2] == (size, sha), "Chunk size or SHA-256 mismatch")


def source_object(item):
    require(isinstance(item, dict), "Invalid source object")
    path = relative(item["path"])
    size = positive(item["size_bytes"])
    git = digest(item["git_blob_sha1"], 40)
    algorithm = item["integrity_algorithm"]
    require(algorithm in ("sha256", "git-blob-sha1"), "Unknown source integrity algorithm")
    sha = item["content_sha256"]
    if sha is not None:
        digest(sha)
    if algorithm == "sha256":
        require(item["lfs"]["size_bytes"] == size and item["lfs"]["sha256"] == sha, "LFS facts disagree")
        digest(sha)
    else:
        require(item["lfs"] is None, "Git file cannot have LFS facts")
    return {"path": path, "size_bytes": size, "git_blob_sha1": git,
            "integrity_algorithm": algorithm, "content_sha256": sha}


def inventory_identity(objects):
    rows = [{"path": x["path"], "size_bytes": x["size_bytes"],
             "git_blob_sha1": x["git_blob_sha1"], "integrity_algorithm": x["integrity_algorithm"],
             "source_sha256": x["content_sha256"] if x["integrity_algorithm"] == "sha256" else None}
            for x in objects]
    return (len(rows), sum(x["size_bytes"] for x in rows),
            hashlib.sha256(encode(sorted(rows, key=lambda x: x["path"]))).hexdigest())


def require_inventory(objects):
    require(inventory_identity(objects) == EXPECTED_INVENTORY,
            "Source file membership or hashes differ from the pinned full repository")


def read_plan(path):
    plan = document(path)
    require(plan["schema_version"] == 1, "Unknown mirror-plan schema")
    source = plan["source"]
    require(source["repo_id"] == REPO and source["revision"] == REVISION
            and source["repo_type"] == "dataset" and source["all_repository_files_included"] is True,
            "Plan must cover the pinned repository")
    objects = [source_object(item) for item in plan["objects"]]
    require(len({x["path"] for x in objects}) == len(objects), "Duplicate source paths")
    require(plan["totals"]["files"] == len(objects)
            and plan["totals"]["bytes"] == sum(x["size_bytes"] for x in objects)
            and len(objects) == 541 and plan["totals"]["bytes"] == 114370281701, "Plan totals disagree with pinned repository")
    require_inventory(objects)
    return {"repo_id": REPO, "revision": REVISION, "license": plan["license"], "objects": objects}


class HuggingFaceSource:
    def __init__(self, token, client=None):
        import httpx
        self.token = token
        self.client = client or httpx.Client(timeout=120, follow_redirects=False)

    @contextmanager
    def worker(self):
        import httpx
        with httpx.Client(timeout=120, follow_redirects=False) as client:
            yield HuggingFaceSource(self.token, client)

    def download(self, item, offset, size, target):
        """Resume only an exact, pinned byte range; redirects never inherit HF auth."""
        start = target.stat().st_size if target.exists() else 0
        require(start <= size, "Oversize partial download")
        if start == size:
            return
        first, last = offset + start, offset + size - 1
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{quote(item['path'], safe='/')}"
        for _ in range(10):
            parsed = urlparse(url)
            require(parsed.scheme == "https" and parsed.hostname, "Non-HTTPS source refused")
            headers = {"Range": f"bytes={first}-{last}", "Accept-Encoding": "identity"}
            if parsed.hostname == "huggingface.co":
                headers["Authorization"] = f"Bearer {self.token}"
            with self.client.stream("GET", url, headers=headers) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    url = urljoin(url, response.headers["location"])
                    continue
                if response.status_code == 200:
                    require(start == 0 and offset == 0 and size == item["size_bytes"],
                            "Source ignored a partial or resumed byte range")
                    require(response.headers.get("content-length") == str(size),
                            "Full source Content-Length mismatch")
                else:
                    require(response.status_code == 206, "Source did not honor byte range")
                    require(response.headers.get("content-range") == f"bytes {first}-{last}/{item['size_bytes']}",
                            "Source Content-Range mismatch")
                require(response.headers.get("content-encoding", "identity") == "identity", "Encoded range refused")
                started = reported = time.monotonic()
                with target.open("ab") as out:
                    remaining = size - start
                    for block in response.iter_raw():
                        now = time.monotonic()
                        require(now - started <= TRANSFER_SECONDS, "Source transfer deadline exceeded")
                        if now - reported >= PROGRESS_SECONDS:
                            progress("download", completed=size-remaining, total=size)
                            reported = now
                        require(len(block) <= remaining, "Source exceeded requested range")
                        out.write(block)
                        remaining -= len(block)
                    out.flush()
                    os.fsync(out.fileno())
                require(remaining == 0, "Truncated source range; resume permitted")
                return
        raise MirrorError("Too many source redirects")


def asset(sha, size):
    from sx_contracts.assets import AssetFormat, AssetRole, AssetRef, AssetProvenance, ProvenancedAsset
    from sx_contracts.content import ContentBlob, Sha256Digest
    return ProvenancedAsset(
        AssetRef(f"urn:bones-seed:sha256:{sha}", ContentBlob(Sha256Digest(sha), size),
                 AssetFormat.OTHER, AssetRole.OTHER, "application/octet-stream", PurePosixPath(f"chunks/{sha}")),
        AssetProvenance(f"https://huggingface.co/datasets/{REPO}", REVISION, f"chunks/{sha}",
                        "LicenseRef-BONES-SEED", "xmimic/tools/mirror_bones_seed.py"))


class CatalogStore:
    """SDK-owned staging, registration, entry membership and verified readback."""
    def __init__(self, catalog, pack, transport=None, catalog_factory=None):
        import httpx
        self.catalog, self.pack = catalog, pack
        self.catalog_factory = catalog_factory
        self.source_path = None
        self.phase = "catalog_staging"
        self._read_grant = None  # one entry, memory only; never journal signed URLs
        self.transport = transport or httpx.Client(timeout=180, follow_redirects=False)

    @contextmanager
    def worker(self):
        import httpx
        require(self.catalog_factory is not None, "Parallel mirror requires isolated Catalog clients")
        with self.catalog_factory() as catalog, httpx.Client(timeout=180, follow_redirects=False) as transport:
            yield CatalogStore(catalog, self.pack, transport, catalog_factory=self.catalog_factory)

    def entry(self, sha):
        return f"{self.pack}.chunks.{sha}"

    def operation_result(self, operation):
        """Reattach only a saved operation's read, never resubmit its registration."""
        from sx_catalog_client.client import CatalogUnavailableError
        self.phase = "catalog_result_read"
        deadline = time.monotonic() + 1800
        last_failure = None
        for attempt, delay in enumerate((0, 2, 5)):
            if delay:
                progress("operation_reattach", attempt=attempt, path=self.source_path, phase=self.phase, **failure_facts(last_failure))
                time.sleep(delay)
            remaining = deadline-time.monotonic()
            require(remaining > 0, "Catalog operation read deadline exceeded")
            try:
                if attempt == 0:
                    return self.catalog.result(operation, timeout_s=remaining)
                with self.catalog_factory() as catalog:
                    return catalog.result(operation, timeout_s=remaining)
            except CatalogUnavailableError as error:
                last_failure = error
                if attempt == 2 or self.catalog_factory is None:
                    raise

    def upload_part(self, stream, upload_id, number, offset, length):
        """Retry only explicit transient responses for identical part bytes/identity."""
        previous = None
        for attempt, delay in enumerate((0, 2, 5)):
            if delay:
                progress("multipart_retry", attempt=attempt, path=self.source_path,
                         phase="catalog_multipart_put", **failure_facts(previous))
                time.sleep(delay)
            # A failed response may have consumed the entire iterator. Rewind and
            # mint a fresh grant without changing the upload ID or part number.
            stream.seek(offset)
            self.phase = "catalog_part_grant"
            grant = self.catalog.staging_upload_part(upload_id, number)
            require(urlparse(grant.url).scheme == "https", "Non-HTTPS upload grant refused")
            def body():
                started = reported = time.monotonic()
                remaining = length
                while remaining:
                    now = time.monotonic()
                    require(now-started <= TRANSFER_SECONDS, "Upload transfer deadline exceeded")
                    if now-reported >= PROGRESS_SECONDS:
                        progress("upload", completed=length-remaining, total=length)
                        reported = now
                    block = stream.read(min(BLOCK, remaining))
                    require(block, "Local upload truncated")
                    remaining -= len(block)
                    yield block
            self.phase = "catalog_multipart_put"
            response = self.transport.put(grant.url, content=body(), headers={"Content-Length": str(length)})
            if not 200 <= response.status_code < 300:
                previous = MultipartTransferError(response.status_code, number)
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    continue
                raise previous
            etag = response.headers.get("etag")
            require(isinstance(etag, str) and etag and len(etag) <= 256, "Missing multipart ETag")
            return etag

    def put(self, path, sha, size, state, save):
        from sx_catalog_client import AssetUpload
        from sx_catalog_client.client import (CatalogOperation, MultipartPart, CompletedStagingUpload,
                                              AbortedStagingUpload)
        from sx_contracts.operations import OperationRef
        self.phase = "catalog_staging"
        check_bytes(path, size, sha)
        declaration = asset(sha, size)
        if "operation" not in state:
            upload = self.catalog.begin_staging_upload(f"bones-seed-{sha}", f"{sha}.chunk")
            require(not isinstance(upload, AbortedStagingUpload), "Staging upload aborted; manual recovery required")
            if "upload_id" in state:
                require(state["upload_id"] == str(upload.upload_id), "Upload identity changed")
            uri = urlparse(upload.source_uri)
            require(uri.scheme in ("s3", "r2") and not uri.query and not uri.fragment,
                    "Catalog staging source must not be a signed URL")
            if "part_size_bytes" in state:
                require(state["part_size_bytes"] == upload.part_size_bytes, "Multipart part size changed")
            state.update(upload_id=str(upload.upload_id), source_uri=upload.source_uri,
                         part_size_bytes=upload.part_size_bytes)
            parts = state.setdefault("parts", {})
            save()
            if isinstance(upload, CompletedStagingUpload):
                require(upload.size_bytes == size, "Completed staging size mismatch")
            else:
                # Never load the SDK's whole part into memory; stream fixed-size pieces.
                with path.open("rb") as stream:
                    for number, offset in enumerate(range(0, size, upload.part_size_bytes), 1):
                        length = min(upload.part_size_bytes, size-offset)
                        if str(number) in parts:
                            continue
                        etag = self.upload_part(stream, upload.upload_id, number, offset, length)
                        parts[str(number)] = etag
                        save()
                self.phase = "catalog_staging_complete"
                staged = self.catalog.complete_staging_upload(upload.upload_id, size_bytes=size,
                    parts=tuple(MultipartPart(int(n), tag) for n, tag in sorted(parts.items(), key=lambda p: int(p[0]))))
                require(staged.source_uri == upload.source_uri, "Completed staging source changed")
                state["source_uri"] = staged.source_uri
                save()
            self.phase = "catalog_registration"
            operation = self.catalog.register(AssetUpload(state["source_uri"], declaration))
            state["operation"] = {"ref": str(operation.ref), "resource_ref": operation.resource_ref}
            save()  # Accepted durable reference before any waiting.
        else:
            operation = CatalogOperation(OperationRef.parse(state["operation"]["ref"]),
                state["operation"]["resource_ref"], AssetUpload(state["source_uri"], declaration))
        require(str(self.operation_result(operation)) == sha, "Catalog result digest mismatch")
        # A quarantined transport entry permits readback before whole-source verification.
        self.phase = "catalog_pack_registration"
        self.catalog.register_asset_pack(self.entry(sha), (declaration,))
        state["registered"] = True
        save()

    def fetch(self, sha, size, target):
        self._read(self.entry(sha), sha, target, expected_size=size)

    def _read(self, entry, sha, target, expected_size=None):
        from sx_contracts.catalog import ObjectReads, RepresentationFormat
        now = time.monotonic()
        cached = self._read_grant
        if cached is not None and cached[0] == entry and now < cached[1]:
            grant = cached[2]
        else:
            grant = self.catalog.access(entry, RepresentationFormat.ASSET_BUNDLE)
            require(isinstance(grant, ObjectReads), "Catalog returned a non-object grant")
            # Reuse the final bundle's grant across chunks; refresh before expiry.
            self._read_grant = (entry, now + grant.expires_in_s - min(30, grant.expires_in_s / 10), grant)
        matches = [obj for obj in grant.objects if str(obj.sha256) == sha]
        require(len(matches) == 1, "Catalog grant does not identify the requested member")
        require(urlparse(matches[0].url).scheme == "https", "Non-HTTPS Catalog grant refused")
        pending = under(target.parent, target.name + ".download")
        with self.transport.stream("GET", matches[0].url, headers={"Accept-Encoding": "identity"}) as response:
            require(response.status_code == 200, "Catalog object download failed")
            require(response.headers.get("content-encoding", "identity") == "identity", "Encoded object refused")
            size = int(response.headers["content-length"])
            require(0 < size <= MAX_CHUNK and (expected_size is None or size == expected_size), "Catalog object size mismatch")
            started = reported = time.monotonic()
            remaining = size
            with pending.open("wb") as stream:
                for block in response.iter_raw():
                    now = time.monotonic()
                    require(now-started <= TRANSFER_SECONDS, "Catalog read deadline exceeded")
                    require(len(block) <= remaining, "Catalog object exceeded declared size")
                    stream.write(block)
                    remaining -= len(block)
                    if now-reported >= PROGRESS_SECONDS:
                        progress("catalog_read", completed=size-remaining, total=size)
                        reported = now
                stream.flush()
                os.fsync(stream.fileno())
            require(remaining == 0, "Truncated Catalog object")
        check_bytes(pending, size, sha)
        os.replace(pending, target)

    def publish(self, manifest, chunks):
        members = {chunk["sha256"]: asset(chunk["sha256"], chunk["size_bytes"]) for chunk in chunks}
        sha = hashlib.sha256(manifest).hexdigest()
        members[sha] = asset(sha, len(manifest))
        self.catalog.register_asset_pack(self.pack, tuple(members.values()))
        return sha

    def read_member(self, sha, target):
        self._read(self.pack, sha, target)
        return target


@contextmanager
def workspace(root):
    import fcntl
    require("raw" not in root.absolute().parts, "Transport scratch cannot be under raw/")
    require(root.absolute() == root.resolve(), "Symlink workspace refused")
    root.mkdir(parents=True, exist_ok=True)
    with under(root, "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def mirror(plan, store, source, output, project, pack, chunk_size=MAX_CHUNK):
    require_inventory(plan["objects"])
    positive(chunk_size)
    require(chunk_size <= MAX_CHUNK, "Chunk exceeds Catalog scratch bound")
    binding = {"plan_sha256": hashlib.sha256(encode(plan)).hexdigest(), "project": project,
               "pack": pack, "chunk_size": chunk_size}
    with workspace(output):
        checkpoint = under(output, "journal.json")
        journal = document(checkpoint) if checkpoint.exists() else {"schema": SCHEMA, "binding": binding, "chunks": {}}
        require(set(journal) == {"schema", "binding", "chunks"} and journal["schema"] == SCHEMA
                and journal["binding"] == binding, "Journal schema or destination changed")
        allowed_keys = {hashlib.sha256(f"{item['path']}:{offset}".encode()).hexdigest()
                        for item in plan["objects"] for offset in range(0, item["size_bytes"], chunk_size)} | {"manifest"}
        require(isinstance(journal["chunks"], dict) and set(journal["chunks"]) <= allowed_keys, "Unknown journal chunk")
        for state in journal["chunks"].values():
            require(isinstance(state, dict) and set(state) <= {"sha256", "upload_id", "source_uri", "part_size_bytes", "parts", "operation", "registered"}, "Unknown journal state")
            if "sha256" in state:
                digest(state["sha256"])
            if "upload_id" in state:
                UUID(state["upload_id"])
            if "part_size_bytes" in state:
                positive(state["part_size_bytes"])
            if "source_uri" in state:
                uri = urlparse(state["source_uri"])
                require(uri.scheme in ("s3", "r2") and not uri.query and not uri.fragment, "Invalid journal staging source")
            if "operation" in state:
                require(set(state["operation"]) == {"ref", "resource_ref"}
                        and state["operation"]["ref"].startswith("experience/asset-registration/")
                        and isinstance(state["operation"]["resource_ref"], str), "Invalid journal operation")
            if "registered" in state:
                require(state["registered"] is True, "Invalid journal completion")
            for number, etag in state.get("parts", {}).items():
                require(re.fullmatch(r"[1-9][0-9]{0,3}|10000", number)
                        and isinstance(etag, str) and 0 < len(etag) <= 256
                        and "://" not in etag and all(32 <= ord(c) < 127 for c in etag), "Invalid multipart checkpoint")
        journal_lock = threading.RLock()
        def save():
            with journal_lock:
                atomic(checkpoint, encode(journal))
        save()
        files = [None] * len(plan["objects"])
        content_locks = {}
        stopped = threading.Event()
        scheduling_lock = threading.Lock()
        pending_files = iter(sorted(enumerate(plan["objects"]),
                                    key=lambda pair: (-pair[1]["size_bytes"], pair[0])))

        def process_file(item, worker_store, worker_source, current):
            whole = hashlib.sha256()
            git = hashlib.sha1(f"blob {item['size_bytes']}\0".encode())
            chunks = []
            for offset in range(0, item["size_bytes"], chunk_size):
                if stopped.is_set():
                    return None
                size = min(chunk_size, item["size_bytes"]-offset)
                key = hashlib.sha256(f"{item['path']}:{offset}".encode()).hexdigest()
                progress("chunk_start", path=item["path"], offset=offset, size_bytes=size)
                with journal_lock:
                    state = copy.deepcopy(journal["chunks"].get(key, {}))
                def save_chunk():
                    with journal_lock:
                        journal["chunks"][key] = copy.deepcopy(state)
                        save()
                local = under(output, f"{key}.chunk")
                if state.get("registered"):
                    sha = digest(state["sha256"])
                    if not local.exists():
                        current["phase"] = "catalog_read"
                        worker_store.fetch(sha, size, local)
                    current["phase"] = "chunk_verify"
                    check_bytes(local, size, sha)
                else:
                    current["phase"] = "source_download"
                    worker_source.download(item, offset, size, local)
                    current["phase"] = "chunk_verify"
                    actual_size, sha, _ = hashes(local)
                    require(actual_size == size, "Source chunk size mismatch")
                    if "sha256" in state:
                        require(state["sha256"] == sha, "Retained chunk changed")
                    state["sha256"] = sha
                    save_chunk()
                    with journal_lock:
                        content_lock = content_locks.setdefault(sha, threading.Lock())
                    # Equal bytes can occur in different files: serialize that one
                    # digest's staging completion, while unrelated content proceeds.
                    with content_lock:
                        current["phase"] = "catalog_registration"
                        worker_store.put(local, sha, size, state, save_chunk)
                current["phase"] = "source_hash"
                with local.open("rb") as stream:
                    while block := stream.read(BLOCK):
                        whole.update(block)
                        git.update(block)
                chunks.append({"offset": offset, "size_bytes": size, "sha256": sha})
                local.unlink()  # private scratch only; Catalog worker independently verified SHA-256
                progress("chunk_registered", path=item["path"], offset=offset, size_bytes=size)
            sha = whole.hexdigest()
            if item["content_sha256"] is not None:
                require(sha == item["content_sha256"], "Whole source SHA-256 mismatch; final pack refused")
            if item["integrity_algorithm"] == "git-blob-sha1":
                require(git.hexdigest() == item["git_blob_sha1"], "Whole Git blob hash mismatch; final pack refused")
            return {**item, "content_sha256": sha, "chunks": chunks}

        def worker():
            current = {"path": None, "phase": "worker_start"}
            try:
                with store.worker() as worker_store, source.worker() as worker_source:
                    while True:
                        with scheduling_lock:
                            if stopped.is_set():
                                return
                            next_file = next(pending_files, None)
                        if next_file is None:
                            return
                        index, item = next_file
                        current.update(path=item["path"], phase="file_start")
                        worker_store.source_path = item["path"]
                        result = process_file(item, worker_store, worker_source, current)
                        with journal_lock:
                            files[index] = result
            except BaseException as error:
                # No new registrations or transfer retries. Other workers retain progress and stop after
                # their in-flight chunk settles, before another file/chunk starts.
                stopped.set()
                if current["phase"] == "catalog_registration":
                    current["phase"] = getattr(worker_store, "phase", current["phase"])
                progress("worker_failed", **failure_facts(error), **current)
                raise

        with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="bones-mirror") as pool:
            futures = [pool.submit(worker) for _ in range(min(WORKERS, len(files)))]
            failures = []
            for future in futures:
                try:
                    future.result()
                except BaseException as error:
                    stopped.set()
                    failures.append(error)
            if failures:
                raise failures[0]
        require(all(item is not None for item in files), "Mirror stopped before all source files verified")
        all_chunks = [chunk for item in files for chunk in item["chunks"]]
        manifest = {"schema": SCHEMA, "source": {"repo_id": REPO, "revision": REVISION},
                    "license": plan["license"], "training_admitted": False, "objects": files}
        content = encode(manifest)
        sha = hashlib.sha256(content).hexdigest()
        local = under(output, "manifest.json")
        atomic(local, content)
        state = journal["chunks"].setdefault("manifest", {})
        store.put(local, sha, len(content), state, save)
        verified = under(output, "manifest.readback")
        store.fetch(sha, len(content), verified)
        check_bytes(verified, len(content), sha)
        verified.unlink()
        require(store.publish(content, all_chunks) == sha, "Published manifest identity mismatch")
        receipt = {"schema": SCHEMA, "project": project, "pack": pack, "manifest_sha256": sha,
                   "files": len(files), "bytes": sum(f["size_bytes"] for f in files)}
        atomic(under(output, "receipt.json"), encode(receipt))
        return receipt


def parse_manifest(value):
    require(set(value) == {"schema", "source", "license", "training_admitted", "objects"}
            and value["schema"] == SCHEMA and value["training_admitted"] is False, "Unknown mirror manifest")
    require(value["source"] == {"repo_id": REPO, "revision": REVISION}, "Unknown source")
    seen = set()
    for item in value["objects"]:
        require(set(item) == {"path", "size_bytes", "git_blob_sha1", "integrity_algorithm", "content_sha256", "chunks"},
                "Unknown object schema")
        name = relative(item["path"])
        require(PurePosixPath(name).parts[0] != ".mirror-restore", "Reserved restore path")
        require(name not in seen, "Duplicate restored path")
        seen.add(name)
        positive(item["size_bytes"])
        digest(item["content_sha256"])
        digest(item["git_blob_sha1"], 40)
        require(item["integrity_algorithm"] in ("sha256", "git-blob-sha1"), "Unknown hash algorithm")
        offset = 0
        for chunk in item["chunks"]:
            require(set(chunk) == {"offset", "size_bytes", "sha256"}, "Unknown chunk schema")
            require(type(chunk["offset"]) is int and chunk["offset"] == offset, "Chunk gap or overlap")
            require(positive(chunk["size_bytes"]) <= MAX_CHUNK, "Oversized chunk")
            digest(chunk["sha256"])
            offset += chunk["size_bytes"]
        require(offset == item["size_bytes"], "Chunk size total mismatch")
    require_inventory(value["objects"])
    return value


def restore(store, sha, output):
    digest(sha)
    with workspace(output / ".mirror-restore"):
        scratch = output / ".mirror-restore"
        local_manifest = under(scratch, "manifest.json")
        store.read_member(sha, local_manifest)
        require(hashes(local_manifest)[1] == sha, "Manifest SHA-256 mismatch")
        manifest = parse_manifest(document(local_manifest))
        checkpoint = under(scratch, "journal.json")
        journal = document(checkpoint) if checkpoint.exists() else {
            "schema": SCHEMA, "manifest_sha256": sha, "files": {}}
        require(set(journal) == {"schema", "manifest_sha256", "files"}
                and journal["schema"] == SCHEMA and journal["manifest_sha256"] == sha,
                "Restore checkpoint belongs to another manifest")
        require(isinstance(journal["files"], dict)
                and set(journal["files"]) <= {x["path"] for x in manifest["objects"]},
                "Unknown restored file in checkpoint")
        def save():
            atomic(checkpoint, encode(journal))
        save()
        for item in manifest["objects"]:
            target = under(output, item["path"])
            completed = journal["files"].get(item["path"], 0)
            require(type(completed) is int and 0 <= completed <= len(item["chunks"]),
                    "Invalid restored chunk boundary")
            if target.exists():
                actual = hashes(target)
                require(actual[:2] == (item["size_bytes"], item["content_sha256"]), "Existing restore differs")
                if item["integrity_algorithm"] == "git-blob-sha1":
                    require(actual[2] == item["git_blob_sha1"], "Existing restored Git blob differs")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            pending = under(scratch, "source-" + hashlib.sha256(item["path"].encode()).hexdigest() + ".partial")
            available = pending.stat().st_size if pending.exists() else 0
            require(available <= item["size_bytes"], "Oversized partial restore")
            boundary, verified = 0, 0
            if pending.exists():
                # Recompute each retained chunk's hash. Recover a completed append that
                # reached disk before its checkpoint; never trust saved hash state.
                with pending.open("rb") as retained:
                    for chunk in item["chunks"]:
                        if available-boundary < chunk["size_bytes"]:
                            break
                        remaining = chunk["size_bytes"]
                        measured = hashlib.sha256()
                        while remaining:
                            block = retained.read(min(BLOCK, remaining))
                            require(block, "Truncated retained restore")
                            remaining -= len(block)
                            measured.update(block)
                        require(measured.hexdigest() == chunk["sha256"], "Retained restore chunk hash mismatch")
                        boundary += chunk["size_bytes"]
                        verified += 1
            require(verified >= completed, "Committed restore bytes are missing")
            journal["files"][item["path"]] = verified
            with pending.open("r+b" if pending.exists() else "w+b") as out:
                out.truncate(boundary)  # discard only the uncommitted incomplete tail
                out.seek(boundary)
                out.flush()
                os.fsync(out.fileno())
                save()
                for index in range(verified, len(item["chunks"])):
                    chunk = item["chunks"][index]
                    part = under(scratch, "chunk")
                    store.read_member(chunk["sha256"], part)
                    check_bytes(part, chunk["size_bytes"], chunk["sha256"])
                    with part.open("rb") as incoming:
                        while block := incoming.read(BLOCK):
                            out.write(block)
                    out.flush()
                    os.fsync(out.fileno())
                    journal["files"][item["path"]] = index+1
                    save()  # Durable chunk boundary before deleting verified scratch.
                    part.unlink()
            actual = hashes(pending)
            require(actual[:2] == (item["size_bytes"], item["content_sha256"]), "Restored source hash mismatch")
            if item["integrity_algorithm"] == "git-blob-sha1":
                require(actual[2] == item["git_blob_sha1"], "Restored Git blob hash mismatch")
            os.replace(pending, target)
            for directory in (target.parent, scratch):
                fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        return len(manifest["objects"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("mirror", "restore"))
    parser.add_argument("--project", required=True, type=UUID)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}", args.pack), "Invalid pack name")
    from sx.credentials import api_key_for
    from sx._transport import ConnectionProfile, Endpoints
    from sx_service.registry import CATALOG
    from sx_auth.principal import ProjectId
    from sx_catalog_client import CatalogClient
    key = os.environ.get("SX_API_KEY") or api_key_for(ConnectionProfile.PRODUCTION)
    require(key, "Existing SDK credential required")
    url = Endpoints.resolve(ConnectionProfile.PRODUCTION).for_door(CATALOG.name)
    def catalog_factory():
        return CatalogClient(url, api_key=key, project_id=ProjectId(str(args.project)))
    with catalog_factory() as catalog:
        store = CatalogStore(catalog, args.pack, catalog_factory=catalog_factory)
        if args.command == "mirror":
            require(args.plan is not None, "mirror requires --plan")
            from huggingface_hub import get_token
            token = get_token()
            require(token, "Existing Hugging Face credential required")
            result = mirror(read_plan(args.plan), store, HuggingFaceSource(token), args.output,
                            str(args.project), args.pack)
        else:
            require(args.manifest_sha256 is not None, "restore requires --manifest-sha256")
            result = {"restored_files": restore(store, args.manifest_sha256, args.output)}
        print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # SDK/HTTP exceptions can contain signed URLs. Do not print their messages or traceback.
        reason = str(error) if isinstance(error, MirrorError) else type(error).__name__
        raise SystemExit(f"Mirror stopped: {reason}; retained checkpoint permits inspection/resume") from None
