"""Inventory every BONES-SEED motion without downloading the motion archives.

Requires an HF account with accepted dataset access (or the pinned local metadata),
plus huggingface_hub and pyarrow. Outputs contain licensed metadata: keep them private.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath

REPO_ID = "bones-studio/seed"
REVISION = "2f59b2077b9da34dd4e43618e705c7cb962c9a66"
METADATA = "metadata/seed_metadata_v004.parquet"
METADATA_SHA256 = "7cbe5624f91ae19af3feee69cff74b2cf553fb26367c8edc4b674c3d8ea97e7d"
EXPECTED_MOTIONS = 142_220
METADATA_FPS = 120
PATH_FIELDS = {
    "soma_uniform": "move_soma_uniform_path",
    "soma_proportional": "move_soma_proportional_path",
    "g1": "move_g1_path",
    "soma_uniform_shape": "move_soma_uniform_shape_path",
    "soma_proportional_shape": "move_soma_proportional_shape_path",
}
# These hashes describe archive objects, not the unpacked individual motion files.
ARCHIVES = {
    "g1.tar.gz": (23_499_973_647, "52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a"),
    "soma_proportional.tar.gz": (45_468_083_856, "f028c26602d6565e706f59e84da0aecfcd05d7348ed1f1f6dec729fd70d32838"),
    "soma_uniform.tar.gz": (45_188_088_297, "0c1fd91628d2644d0d7c0b735ae64faa5fa79834280bc1ce7860a871e55f912c"),
}
TEMPORAL_LABELS = {
    "path": "metadata/seed_metadata_v002_temporal_labels.jsonl",
    "size_bytes": 58_447_630,
    "sha256": "302da3c319bbed4f712fa22a4d71b5288493911140cd72579e29de2ac324d634",
    "join_key": "filename",
    "status": "not_downloaded_or_joined",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Source paths must be nonempty relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise ValueError(f"Unsafe or noncanonical source path: {value!r}")
    return value


def inventory(rows: list[dict], output: Path, local_root: Path | None = None) -> dict:
    """Preserve all metadata rows; identify mirror families before any later split.

    A local-root check establishes file presence only, never byte identity or quality.
    No training split or retarget admission is performed here.
    """
    names: dict[str, dict] = {}
    for row in rows:
        name = row.get("filename")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"Missing or duplicate filename: {name!r}")
        if not isinstance(row.get("is_mirror"), bool):
            raise ValueError(f"{name}: is_mirror must be boolean")
        if type(row.get("move_duration_frames")) is not int or row["move_duration_frames"] <= 0:
            raise ValueError(f"{name}: expected a positive integer frame count")
        for field in ("actor_uid", "take_name", "package", "category"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"{name}: missing {field}")
        if type(row.get("take_date")) is not int:
            raise ValueError(f"{name}: take_date must be an integer")
        for field in PATH_FIELDS.values():
            source_path(row.get(field))
        names[name] = row

    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    packages = Counter()
    categories = Counter()
    actors = set()
    groups = set()
    unresolved = []
    missing = Counter()
    total_frames = 0
    root = local_root.resolve() if local_root is not None else None
    presence_cache: dict[str, bool] = {}
    motions = output / "motions.jsonl.gz"
    # mtime=0 and no embedded filename make repeated identical inventories byte-identical.
    with motions.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
        for name, row in sorted(names.items()):
            mirror = row["is_mirror"]
            original = name[:-2] if mirror and name.endswith("_M") else name
            counterpart = names.get(original)
            group_valid = not mirror or (
                name.endswith("_M") and counterpart is not None and not counterpart["is_mirror"]
                and all(row[k] == counterpart[k] for k in ("actor_uid", "take_name", "take_date"))
            )
            if not group_valid:
                unresolved.append(name)
            else:
                groups.add(original)
            paths = {kind: row[field] for kind, field in PATH_FIELDS.items()}
            presence = None
            if root is not None:
                presence = {}
                for kind, relative in paths.items():
                    if relative not in presence_cache:
                        path = (root / relative).resolve()
                        if not path.is_relative_to(root):
                            raise ValueError(f"Source path escapes local root: {relative}")
                        presence_cache[relative] = path.is_file()
                    presence[kind] = presence_cache[relative]
                    if not presence[kind]:
                        missing[kind] += 1
            entry = {
                "filename": name,
                "source_group": original if group_valid else None,
                "metadata": row,
                "local_file_presence": presence,
            }
            stream.write((json.dumps(entry, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode())
            counts["mirrored" if mirror else "original"] += 1
            packages[row["package"]] += 1
            categories[row["category"]] += 1
            actors.add(row["actor_uid"])
            total_frames += row["move_duration_frames"]
    summary = {
        "motions": len(rows),
        "original": counts["original"],
        "mirrored": counts["mirrored"],
        "source_groups": len(groups),
        "unresolved_mirror_filenames": unresolved,
        "actors": len(actors),
        "packages": dict(sorted(packages.items())),
        "categories": dict(sorted(categories.items())),
        "metadata_frames": total_frames,
        "metadata_fps": METADATA_FPS,
        "metadata_frame_hours": total_frames / METADATA_FPS / 3600,
        "local_presence": {
            "status": "checked_presence_only" if root is not None else "not_inspected",
            "root": str(root) if root is not None else None,
            "missing_motion_references": {kind: missing[kind] for kind in PATH_FIELDS} if root is not None else None,
        },
        "motion_manifest": {"path": motions.name, "sha256": sha256(motions)},
        "retargeted_motions": 0,
        "training_admitted_motions": 0,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, help="Use an already downloaded copy of the pinned parquet")
    parser.add_argument("--output", type=Path, default=Path("data/bones_seed/inventory"))
    parser.add_argument("--local-root", type=Path, help="Check source-file presence in an existing extracted dataset")
    args = parser.parse_args()
    if args.local_root is not None and not args.local_root.is_dir():
        parser.error("--local-root must be an existing extracted dataset directory")
    metadata = args.metadata
    if metadata is None:
        from huggingface_hub import hf_hub_download

        metadata = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", revision=REVISION, filename=METADATA))
    digest = sha256(metadata)
    if digest != METADATA_SHA256:
        raise ValueError(f"Metadata digest differs from pinned v004: {digest}")
    import pyarrow.parquet as pq

    table = pq.read_table(metadata)
    if table.num_rows != EXPECTED_MOTIONS or table.num_columns != 51:
        raise ValueError("Pinned metadata must contain all 142220 motions and 51 columns")
    summary = inventory(table.to_pylist(), args.output, args.local_root)
    manifest = {
        "source": {"repo_id": REPO_ID, "revision": REVISION, "metadata_path": METADATA, "metadata_sha256": digest},
        "status": "metadata_inventory_only",
        "license_url": "https://bones.studio/info/seed-license",
        "archives": [{"path": path, "size_bytes": size, "sha256": digest, "status": "not_downloaded_or_verified"}
                     for path, (size, digest) in ARCHIVES.items()],
        "temporal_labels": TEMPORAL_LABELS,
        **summary,
    }
    destination = args.output / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(destination), "motions": summary["motions"],
                      "original": summary["original"], "mirrored": summary["mirrored"],
                      "unresolved_mirror_groups": len(summary["unresolved_mirror_filenames"]),
                      "status": manifest["status"]}, indent=2))


if __name__ == "__main__":
    main()
