#!/usr/bin/env python3
"""Build the owning SDK wheel and retain exact source/file identities for runtime images.

Run with Python 3.12+ and uv on PATH. This never builds or publishes a container.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "assets",
    "_assets",
    "build",
    "dist",
    "target",
    ".pytest_cache",
    ".ruff_cache",
}
SUFFIXES = {
    ".py",
    ".pyi",
    ".proto",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".md",
    ".typed",
}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_files(root: Path, directories: list[Path]) -> dict[str, str]:
    result = {}
    for directory in directories:
        for path in sorted(directory.rglob("*")) if directory.is_dir() else [directory]:
            relative = path.relative_to(root)
            if set(relative.parts) & SKIP or (
                directory.is_dir() and path.suffix not in SUFFIXES
            ):
                continue
            if path.is_symlink():
                raise ValueError(f"runtime source must not be a symlink: {relative}")
            if path.is_file():
                result[relative.as_posix()] = sha(path)
    return dict(sorted(result.items()))


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def verify(directory: Path, *, source: Path = ROOT) -> dict:
    manifest = json.loads((directory / "manifest.json").read_bytes())
    for name, digest in manifest["xmimic_files"].items():
        if sha(source / name) != digest:
            raise ValueError(f"XMimic source changed after runtime snapshot: {name}")
    for name, digest in manifest["wheels"].items():
        if sha(directory / "wheels" / name) != digest:
            raise ValueError(f"runtime wheel differs from retained manifest: {name}")
    for name, digest in manifest.get("dependency_locks", {}).items():
        if sha(directory / name) != digest:
            raise ValueError(f"dependency lock differs from retained manifest: {name}")
    with zipfile.ZipFile(
        directory / "wheels" / next(iter(manifest["wheels"]))
    ) as wheel:
        names = wheel.namelist()
        for package in ("sx_training", "sx_embodiments", "sx_geometry"):
            if not any(name.startswith(package + "/") for name in names):
                raise ValueError(f"runtime wheel misses {package}")
        if any(name.startswith("sx_embodiments/_assets/") for name in names):
            raise ValueError("embodiment assets must arrive as governed inputs")
    return manifest


def dependency_locks(stage: Path, wheel: Path) -> dict[str, str]:
    """Resolve binary Linux wheels and keep hashes, including our exact local SDK."""
    result = {}
    for kind in ("cpu", "gpu"):
        requirements = (ROOT / "requirements.txt").read_text().splitlines()
        if kind == "cpu":
            requirements = [
                line
                for line in requirements
                if line.split("==", 1)[0] in {"jax", "jaxlib", "numpy", "mujoco"}
            ]
        request = stage / f"{kind}.in"
        request.write_text(str(wheel) + "[training]\n" + "\n".join(requirements) + "\n")
        output = stage / f"{kind}.lock"
        subprocess.run(
            [
                "uv",
                "pip",
                "compile",
                str(request),
                "--python-version",
                "3.12",
                "--python-platform",
                "x86_64-manylinux_2_28",
                "--only-binary",
                ":all:",
                "--generate-hashes",
                "--quiet",
                "-o",
                str(output),
            ],
            check=True,
        )
        text = output.read_text().replace(
            str(wheel), "/opt/runtime-wheels/wheels/" + wheel.name
        )
        text = (
            "\n".join(
                line
                for line in text.splitlines()
                if not line.startswith("#    uv pip compile")
            )
            + "\n"
        )
        output.write_text(text)
        result[output.name] = sha(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--owner",
        type=Path,
        help="exact SentientX owner checkout with generated protocol",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/runtime-images")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.verify:
        verify(output)
        print("Verified source snapshot and local runtime wheel hashes")
        return
    if args.owner is None:
        parser.error("--owner is required when building")
    owner = args.owner.resolve()
    sys.path.insert(0, str(owner / "packages/sx-contracts"))
    from sx_contracts.identity import canonical_json, content_digest

    builder_path = owner / "packages/sx/scripts/build_sdk_dist.py"
    description = importlib.util.spec_from_file_location(
        "_runtime_sdk_builder", builder_path
    )
    if description is None or description.loader is None:
        raise ValueError("owning SDK builder is unavailable")
    builder = importlib.util.module_from_spec(description)
    sys.modules[description.name] = builder
    description.loader.exec_module(builder)
    members = builder.discover()
    optional = tuple(members["sx"].config["project"].get("optional-dependencies", {}))
    closure = builder.resolve(members, "sx", optional)
    directories = [member.directory for member in closure.members] + [
        builder_path,
        owner / "pyproject.toml",
        owner / "uv.lock",
    ]
    sdk_files = source_files(owner, directories)
    sdk_sha = str(content_digest(sdk_files))
    previous = (
        json.loads((output / "manifest.json").read_bytes())
        if (output / "manifest.json").exists()
        else None
    )
    if previous is None or previous["sdk_source_sha256"] != sdk_sha:
        version = "0.0.0+src." + sdk_sha[:24]
        subprocess.run(
            [sys.executable, str(builder_path), "--version", version],
            cwd=owner,
            check=True,
        )
        if source_files(owner, directories) != sdk_files:
            raise ValueError(
                "owner source changed during wheel build; retain output and reconcile before use"
            )
        wheels = list((owner / "build/sdk/dist").glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("owning SDK builder did not produce exactly one wheel")
    else:
        wheels = [
            output / "wheels" / name
            for name, digest in previous["wheels"].items()
            if sha(output / "wheels" / name) == digest
        ]
        if len(wheels) != 1:
            raise ValueError("retained runtime wheel is missing or corrupt")
    fork_files = source_files(
        ROOT,
        [
            ROOT / "xmimic",
            ROOT / "requirements.txt",
            ROOT / "Dockerfile.cpu",
            ROOT / "Dockerfile.gpu",
            ROOT / ".dockerignore",
            Path(__file__),
        ],
    )
    manifest = {
        "schema": "xmimic.runtime-images/v1",
        "sdk_head": git_head(owner),
        "embodiment_head": git_head(owner / "packages/sx-embodiments"),
        "sdk_source_sha256": sdk_sha,
        "sdk_files": sdk_files,
        "xmimic_head": git_head(ROOT),
        "xmimic_source_sha256": str(content_digest(fork_files)),
        "xmimic_files": fork_files,
        "wheels": {wheel.name: sha(wheel) for wheel in wheels},
        "sdk_closure": [member.dist for member in closure.members],
        "provenance": "Exact working-tree file hashes, including uncommitted code; HEAD is context only",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="runtime-wheels-", dir=output.parent
    ) as temporary:
        stage = Path(temporary)
        (stage / "wheels").mkdir()
        for wheel in wheels:
            shutil.copy2(wheel, stage / "wheels" / wheel.name)
        manifest["dependency_locks"] = dependency_locks(
            stage, next((stage / "wheels").glob("*.whl"))
        )
        (stage / "manifest.json").write_text(canonical_json(manifest))
        verify(stage)
        output.mkdir(exist_ok=True)
        (output / "wheels").mkdir(exist_ok=True)
        for wheel in (stage / "wheels").iterdir():
            shutil.copy2(wheel, output / "wheels" / wheel.name)
        for lock in stage.glob("*.lock"):
            shutil.copy2(lock, output / lock.name)
        (stage / "manifest.json").replace(output / "manifest.json")
    print(f"Prepared {output}; SDK source {sdk_sha}; no image built or published")


if __name__ == "__main__":
    main()
