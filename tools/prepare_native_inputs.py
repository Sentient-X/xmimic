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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", choices=list(NativeMethod), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish-project")
    parser.add_argument("--producer-revision", required=True)
    args = parser.parse_args()
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
