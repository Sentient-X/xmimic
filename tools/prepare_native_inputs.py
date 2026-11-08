"""Prepare/publish native controls; launch only with the existing SentientX training CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from sx_contracts import Sha256Digest
from sx_contracts.identity import canonical_json
from sx_training.artifacts import (
    ArtifactFile,
    ArtifactManifest,
    ArtifactPurpose,
    verified_artifacts,
)
from tools.benchmark_native import NativeMethod, SOURCES, archived_files, digest
from xmimic.artifact_cache import CatalogPublisher, Provenance


def prepare(
    root: Path, method: NativeMethod, output: Path
) -> tuple[str, ArtifactManifest]:
    selected = next(item for item in SOURCES if item.method is method)
    archive = root / (selected.directory.lower() + ".tar.gz")
    if archive.is_symlink() or digest(archive) != selected.archive_sha256:
        raise ValueError("Native source archive differs")
    members = {"source.tar.gz": archive}
    source_files = archived_files(archive, selected.directory)
    provenance = {
        "source": {
            "repository": selected.repository,
            "revision": selected.revision,
            "archive_sha256": selected.archive_sha256,
        }
    }
    if method is NativeMethod.XMIMIC:
        tasks = json.loads((root / "xmimic-task.json").read_bytes())
        if len(tasks["files"]) != 75:
            raise ValueError("Native XMimic requires its complete75-file task closure")
        source_manifest = source_files["tools/data_sha256.txt"].decode()
        expected = {
            row.split(maxsplit=1)[1].lstrip(" *"): row.split(maxsplit=1)[0]
            for row in source_manifest.splitlines()
        }
        if tasks["files"].keys() != expected.keys():
            raise ValueError("Native task inventory differs from upstream")
        for name, facts in tasks["files"].items():
            path = root / "xmimic-task" / name
            if (
                path.is_symlink()
                or digest(path) != expected[name]
                or facts["sha256"] != expected[name]
            ):
                raise ValueError("Native task bytes differ from upstream")
            members["task/" + name] = path
        provenance["tasks"] = tasks
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "native-acquisition.json"
    payload = canonical_json(provenance).encode()
    if provenance_path.exists() and provenance_path.read_bytes() != payload:
        raise ValueError("Native acquisition identity changed")
    if not provenance_path.exists():
        provenance_path.write_bytes(payload)
    members["native-acquisition.json"] = provenance_path
    inputs = output / "inputs"
    if inputs.is_symlink():
        raise ValueError("Linked native input directory")
    inputs.mkdir(exist_ok=True)
    for name, path in members.items():
        target = inputs / name
        if any(parent.is_symlink() for parent in (target, *target.parents)):
            raise ValueError("Linked native input path")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or digest(target) != digest(path):
                raise ValueError("Retained native input changed")
        else:
            os.link(path, target)
    actual = {
        p.relative_to(inputs).as_posix()
        for p in inputs.rglob("*")
        if p.is_file() or p.is_symlink()
    }
    if actual != members.keys():
        raise ValueError("Unrecorded native inputs")
    training = method is NativeMethod.XMIMIC
    manifest = ArtifactManifest(
        ArtifactPurpose.TRAINING if training else ArtifactPurpose.EVALUATION,
        tuple(
            ArtifactFile(name, Sha256Digest(digest(path)), path.stat().st_size)
            for name, path in sorted(members.items())
        ),
        training,
    )
    payload = manifest.canonical_bytes()
    manifest_path = output / "artifact-manifest.json"
    if manifest_path.exists() and manifest_path.read_bytes() != payload:
        raise ValueError("Native manifest identity changed")
    if not manifest_path.exists():
        manifest_path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), manifest


def evaluation_manifest(source, output, provenance, receipt, *, seed):
    """Compose references only; a completed native command is not an evaluation score."""
    selected = next(item for item in SOURCES if item.method is NativeMethod.XMIMIC)
    source_files = {item.path: item for item in source.files}
    output_files = {item.path: item for item in output.files}
    if (
        source.purpose is not ArtifactPurpose.TRAINING
        or not source.training_eligible
        or output.purpose is not ArtifactPurpose.TRANSFORM
        or not output.training_eligible
        or len(source_files) != 77
        or sum(name.startswith("task/") for name in source_files) != 75
        or source_files.get("source.tar.gz") is None
        or str(source_files["source.tar.gz"].sha256) != selected.archive_sha256
    ):
        raise ValueError("Native training parent closure differs")
    for key, value in {
        "method": NativeMethod.XMIMIC.value,
        "phase": "train",
        "seed": seed,
        "repository": selected.repository,
        "revision": selected.revision,
        "source_sha256": selected.archive_sha256,
    }.items():
        if provenance.get(key) != value:
            raise ValueError("Native training provenance differs")
    steps = provenance.get("steps")
    if (
        not isinstance(steps, list)
        or len(steps) != 1
        or steps[0].get("state") != "complete"
    ):
        raise ValueError("Native training command is not complete")
    if receipt.get("files") != {
        name: str(item.sha256)
        for name, item in output_files.items()
        if name != "training-receipt.json"
    }:
        raise ValueError("Owner receipt differs from output manifest")
    from sx_training import SCHEMA

    if receipt.get("schema") != SCHEMA:
        raise ValueError("Unknown owner training receipt schema")
    for field in ("input_sha256", "spec_sha256"):
        Sha256Digest(receipt[field])
    checkpoints = [item for item in output.files if item.path.startswith("ckpt_")]
    if {item.path for item in checkpoints} != {
        f"ckpt_{n:06d}.npz" for n in range(100, 3501, 100)
    } or any(
        "/" in item.path or not item.path.endswith(".npz") for item in checkpoints
    ):
        raise ValueError("Native output has no exact checkpoint inventory")
    if steps[0].get("files") != {
        name: str(item.sha256)
        for name, item in output_files.items()
        if name not in {"native-provenance.json", "training-receipt.json"}
    }:
        raise ValueError("Native command receipt differs from output manifest")
    files = list(source.files)
    files.extend(
        ArtifactFile("checkpoints/" + item.path, item.sha256, item.size_bytes)
        for item in checkpoints
    )
    # Retain proof by reference too, without copying checkpoint bytes.
    files.extend(
        ArtifactFile(
            "training/" + name, output_files[name].sha256, output_files[name].size_bytes
        )
        for name in ("native-provenance.json", "training-receipt.json")
    )
    return ArtifactManifest(
        ArtifactPurpose.EVALUATION, tuple(sorted(files, key=lambda x: x.path)), False
    )


def compose_evaluation(
    catalog,
    publisher,
    *,
    job,
    result,
    source_provenance,
    directory,
    read_pack,
    read_lineage,
):
    """Publish a small manifest and bind existing promoted assets, never their bytes.

    read_pack and read_lineage use the existing public Catalog REST/generated RPC
    read surfaces. A lost mutation acknowledgement is reconciled against those
    surfaces before any new effect. Unknown state refuses instead of replaying.
    """
    import fcntl
    from dataclasses import asdict
    from sx_training.artifacts import artifact_manifest_from_bytes
    from sentientx.sx.autonomy.train.v1 import training_pb2
    from sentientx.sx.experience.catalog.v1 import sources_pb2
    from xmimic.artifact_cache import _atomic, content_hash

    if job.status != training_pb2.TRAINING_JOB_STATUS_SUCCEEDED:
        raise ValueError("Native training Job must have succeeded")
    config = json.loads(job.authored_trainer.config_json)
    if (
        job.authored_trainer.entrypoint != "tools.benchmark_native:run"
        or config
        != {"method": "vanilla_xmimic", "phase": "train", "seed": job.execution.seed}
        or job.authored_trainer.steps != 3500
    ):
        raise ValueError("Job is not the declared original native training protocol")
    source_entry = job.artifact_inputs.entry
    source_sha = job.artifact_inputs.manifest_sha256
    output_entry = f"training.artifacts.{job.id}"
    output_sha = str(result.sha256)
    binding = {
        "project": publisher.project_id,
        "job": job.id,
        "source": [source_entry, source_sha],
        "output": [output_entry, output_sha],
        "producer": asdict(publisher.provenance),
        "source_provenance": asdict(source_provenance),
    }
    if any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ValueError("Linked composition directory")
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "composition.lock"
    if lock.is_symlink():
        raise ValueError("Linked composition lock")
    with lock.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        journal = directory / "composition.json"
        state = {"binding": binding, "phase": "prepared"}
        if journal.exists():
            if journal.is_symlink():
                raise ValueError("Linked composition journal")
            raw = json.loads(journal.read_bytes())
            state = raw["document"]
            if raw["sha256"] != content_hash(state) or state["binding"] != binding:
                raise ValueError("Composition identity changed or journal corrupt")
            if set(state) != {"binding", "phase"} or state["phase"] not in {
                "prepared",
                "manifest",
                "registering",
                "registered",
                "lineage",
                "complete",
            }:
                raise ValueError("Unknown composition journal state")

        def save(phase):
            state["phase"] = phase
            _atomic(
                journal,
                canonical_json(
                    {"sha256": content_hash(state), "document": state}
                ).encode(),
            )

        save(state["phase"])

        metadata = {}
        for parent in (source_entry, output_entry):
            pack = read_pack(parent)
            if pack is None or pack["name"] != parent:
                raise ValueError("Exact parent pack is not readable")
            rows = pack["files"]
            if any(
                row["role"] != "other"
                or type(row["size_bytes"]) is not int
                or row["size_bytes"] <= 0
                for row in rows
            ):
                raise ValueError("Parent asset role or size differs")
            metadata[parent] = {row["sha256"]: row for row in rows}
            if len(metadata[parent]) != len(rows):
                raise ValueError("Ambiguous duplicate parent references")

        # Only manifests and two small proof documents are read, never checkpoints.
        def small(entry, sha, expected_size=None):
            Sha256Digest(sha)
            size = metadata[entry][sha]["size_bytes"]
            if size > 1024**2 or (expected_size is not None and size != expected_size):
                raise ValueError("Native metadata size differs or exceeds 1 MiB")
            path = directory / (sha + ".json")
            if path.is_symlink():
                raise ValueError("Linked metadata file")
            if not path.exists():
                # Catalog grants custody; this SDK method verifies the received SHA.
                catalog.download_bundle_member(entry, sha, path)
            payload = path.read_bytes()
            if len(payload) != size or hashlib.sha256(payload).hexdigest() != sha:
                raise ValueError("Native metadata content differs")
            return payload

        source = artifact_manifest_from_bytes(small(source_entry, source_sha))
        output = artifact_manifest_from_bytes(small(output_entry, output_sha))
        for parent, declared, manifest_sha in (
            (source_entry, source, source_sha),
            (output_entry, output, output_sha),
        ):
            expected = {str(item.sha256): item.size_bytes for item in declared.files}
            expected[manifest_sha] = metadata[parent][manifest_sha]["size_bytes"]
            if {
                sha: row["size_bytes"] for sha, row in metadata[parent].items()
            } != expected:
                raise ValueError(
                    "Parent manifest differs from registered hash/size inventory"
                )
        output_files = {item.path: item for item in output.files}
        if source_entry not in read_lineage(output_entry):
            raise ValueError("Training output has no recorded input ancestry")
        proof = {}
        for name in ("native-provenance.json", "training-receipt.json"):
            item = output_files[name]
            proof[name] = json.loads(
                small(output_entry, str(item.sha256), item.size_bytes)
            )
        manifest = evaluation_manifest(
            source,
            output,
            proof["native-provenance.json"],
            proof["training-receipt.json"],
            seed=job.execution.seed,
        )
        payload = manifest.canonical_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        entry = f"native.vanilla_xmimic.eval.{sha[:24]}"
        path = directory / "artifact-manifest.json"
        if path.exists() and (path.is_symlink() or path.read_bytes() != payload):
            raise ValueError("Evaluation manifest changed")
        _atomic(path, payload)
        origins = {
            str(item.sha256): (source_entry, item.size_bytes) for item in source.files
        }
        for item in manifest.files:
            origins.setdefault(str(item.sha256), (output_entry, item.size_bytes))

        # Use the actual owner-returned references, including original URI/path/role.
        # Registration binds references only and never rewrites byte provenance.
        from concurrent.futures import ThreadPoolExecutor

        def existing(item):
            digest, (parent, size) = item
            ref = catalog.read_asset(parent, Sha256Digest(digest)).asset
            row = metadata[parent][digest]
            if (
                str(ref.sha256) != digest
                or ref.content.size_bytes != size
                or ref.role.value != row["role"]
                or (None if ref.logical_path is None else str(ref.logical_path))
                != row["path"]
            ):
                raise ValueError("Parent AssetRef differs from registered metadata")
            return ref

        with ThreadPoolExecutor(max_workers=8) as pool:
            assets = list(pool.map(existing, origins.items()))
        # The only bytes promoted by this handoff are the new manifest.
        if state["phase"] == "prepared":
            publisher.publish_files(
                {sha: path},
                path,
                directory / "manifest-publication",
                entry + ".manifest",
            )
            save("manifest")
        assets.append(publisher._asset(sha, len(payload)).asset)
        expected = {
            (
                str(a.sha256),
                a.content.size_bytes,
                a.role.value,
                None if a.logical_path is None else str(a.logical_path),
            )
            for a in assets
        }

        def pack_matches():
            pack = read_pack(entry)
            if pack is None:
                return False
            if (
                pack["name"] != entry
                or {
                    (f["sha256"], f["size_bytes"], f["role"], f["path"])
                    for f in pack["files"]
                }
                != expected
            ):
                raise ValueError("Registered evaluation pack membership differs")
            return True

        if state["phase"] == "registering":
            if not pack_matches():
                raise RuntimeError(
                    "Ambiguous pack registration; reconcile before resubmission"
                )
            save("registered")
        if state["phase"] == "manifest":
            save("registering")
            catalog.register_asset_pack(entry, tuple(assets))
            if not pack_matches():
                raise ValueError("Evaluation pack was not readable after registration")
            save("registered")
        parents = {source_entry, output_entry}
        if state["phase"] == "lineage":
            if not parents <= read_lineage(entry):
                raise RuntimeError(
                    "Ambiguous lineage acknowledgement; reconcile before resubmission"
                )
            save("complete")
        if state["phase"] == "registered":
            save("lineage")
            catalog.record_source_lineage(
                sources_pb2.RecordLineageRequest(
                    entry=entry,
                    parents=[
                        sources_pb2.LineageParent(
                            entry=p, supplied=["LINEAGE_SUPPLIED_REFERENCE"]
                        )
                        for p in sorted(parents)
                    ],
                )
            )
            if not parents <= read_lineage(entry):
                raise ValueError("Evaluation parent lineage was not readable")
            save("complete")
        if not pack_matches() or not parents <= read_lineage(entry):
            raise ValueError("Completed evaluation composition no longer matches")
        return {
            "entry": entry,
            "manifest_sha256": sha,
            "purpose": "evaluation",
            "training_eligible": False,
            "job": job.id,
            "parents": sorted(parents),
            "files": len(manifest.files),
            "promoted_bytes": len(payload),
        }


def compose_from_live_job(
    catalog, publisher, *, url, key, job_id, source_output, expected_source
):
    """Use existing signed-in authority and public owner reads; never retain credentials."""
    import httpx
    from sentientx.sx.autonomy.train.v1 import training_connect, training_pb2
    from sentientx.sx.common.v1 import common_pb2
    from sentientx.sx.experience.catalog.v1 import sources_connect, sources_pb2
    from sx._transport import ConnectionProfile, Endpoints
    from sx_service.registry import AUTONOMY
    from xmimic.artifact_cache import content_hash

    headers = {"X-API-Key": key, "X-SX-Project-Id": publisher.project_id}
    training_url = Endpoints.resolve(ConnectionProfile.PRODUCTION).for_door(
        AUTONOMY.name
    )
    with training_connect.TrainingJobServiceClientSync(training_url) as training:
        job = training.get_training_job(
            training_pb2.GetTrainingJobRequest(job_id=job_id), headers=headers
        ).job
        if (
            job.id != job_id
            or (job.artifact_inputs.entry, job.artifact_inputs.manifest_sha256)
            != expected_source
        ):
            raise ValueError("Training Job differs from the declared input parent")
        response = training.get_training_operation_result(
            training_pb2.GetTrainingOperationResultRequest(
                operation=common_pb2.OperationRef(
                    pillar=common_pb2.PLATFORM_PILLAR_AUTONOMY,
                    kind=common_pb2.OPERATION_KIND_TRAINING,
                    operation_id=job_id,
                )
            ),
            headers=headers,
        )
    product = response.result.training
    if not product.HasField("artifact"):
        raise ValueError("Training Job did not publish an authored artifact")
    raw = json.loads((source_output / "publication/publication.json").read_bytes())
    if content_hash(raw["document"]) != raw["sha256"]:
        raise ValueError("Original publication journal is corrupt")
    binding = raw["document"]["binding"]
    if (binding["pack"], binding["manifest_sha256"]) != expected_source or binding[
        "project_id"
    ] != publisher.project_id:
        raise ValueError("Original publication belongs to another parent/project")
    origin = Provenance(**binding["provenance"])
    with (
        httpx.Client(base_url=url, headers=headers, timeout=30) as http,
        sources_connect.CatalogSourceServiceClientSync(url) as sources,
    ):

        def read_pack(name):
            result = http.get("/api/asset-packs/" + name)
            if result.status_code == 404:
                return None
            result.raise_for_status()
            return result.json()

        def read_lineage(name):
            request = sources_pb2.GetLineageRequest(entry=name, depth=1)
            result = sources.get_lineage(request, headers=headers)
            if (
                result.request != request
                or result.subject.label != name
                or result.truncated
            ):
                raise ValueError("Catalog lineage response differs or is incomplete")
            labels = {node.ref: node.label for node in result.nodes}
            return {
                labels[edge.parent]
                for edge in result.edges
                if edge.child == result.subject.ref and 4 in edge.supplied
            }

        return compose_evaluation(
            catalog,
            publisher,
            job=job,
            result=product.artifact,
            source_provenance=origin,
            directory=source_output / ("evaluation-" + job_id),
            read_pack=read_pack,
            read_lineage=read_lineage,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", choices=list(NativeMethod), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish-project")
    parser.add_argument(
        "--training-job-id", help="Compose evaluation from this succeeded native Job"
    )
    parser.add_argument("--producer-revision", required=True)
    args = parser.parse_args()
    if args.training_job_id and (
        not args.publish_project or args.method != NativeMethod.XMIMIC
    ):
        parser.error("--training-job-id requires XMimic and --publish-project")
    sha, manifest = prepare(
        args.root.resolve(), NativeMethod(args.method), args.output.resolve()
    )
    entry = f"native.{args.method}.{sha[:24]}"
    if args.publish_project:
        from sx.credentials import api_key_for
        from sx._transport import ConnectionProfile, Endpoints
        from sx_service.registry import CATALOG
        from sx_auth.principal import ProjectId
        from sx_catalog_client import CatalogClient

        key = api_key_for(ConnectionProfile.PRODUCTION)
        if not key:
            raise ValueError("An existing signed-in profile is required")
        url = Endpoints.resolve(ConnectionProfile.PRODUCTION).for_door(CATALOG.name)
        named = verified_artifacts(manifest, args.output / "inputs")
        files = {str(item.sha256): named[item.path] for item in manifest.files}
        path = args.output / "artifact-manifest.json"
        files[sha] = path
        with CatalogClient(
            url, api_key=key, project_id=ProjectId(args.publish_project)
        ) as catalog:
            publisher = CatalogPublisher(
                catalog,
                args.publish_project,
                Provenance(
                    "https://github.com/Sentient-X/xmimic",
                    args.producer_revision,
                    "LicenseRef-Upstream-Native-Baseline",
                    "xmimic.native-input-closure",
                ),
            )
            if args.training_job_id:
                composed = compose_from_live_job(
                    catalog,
                    publisher,
                    url=url,
                    key=key,
                    job_id=args.training_job_id,
                    source_output=args.output,
                    expected_source=(entry, sha),
                )
                print(json.dumps(composed))
                return
            publisher.publish_files(files, path, args.output / "publication", entry)
    print(
        json.dumps(
            {
                "entry": entry,
                "manifest_sha256": sha,
                "purpose": manifest.purpose.value,
                "training_eligible": manifest.training_eligible,
                "files": len(manifest.files),
                "published": bool(args.publish_project),
            }
        )
    )


if __name__ == "__main__":
    main()
