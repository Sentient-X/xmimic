"""Extract a named selection of BONES-SEED SOMA clips into a Modal Volume.

The source archive is one 45 GB gzip tar (not seekable), so the only way to take a
few hundred clips out of it is one streaming pass. This runs that pass next to the
source on Modal CPU (no GPU), keeps only the selected members, and writes them to
the ``bones-seed-clips`` volume with a receipt of every requested name: found
members with their SHA-256, and missing ones listed rather than dropped.

    modal run tools/modal_extract_bones_clips.py \
        --selection data/bones_seed/demo-pool/selection.jsonl

Needs the Modal secret ``huggingface`` with an ``HF_TOKEN`` that holds approved
``bones-studio/seed`` access. The revision is pinned; bytes are never re-published.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = "bones-studio/seed"
REVISION = "2f59b2077b9da34dd4e43618e705c7cb962c9a66"
ARCHIVE = "soma_uniform.tar.gz"

app = modal.App("xmimic-bones-extract")
volume = modal.Volume.from_name("bones-seed-clips", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface-hub", "requests")


@app.function(
    image=image,
    volumes={"/clips": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3 * 3600,
    cpu=2.0,
    memory=4096,
)
def extract(names: list[str], tag: str) -> dict[str, object]:
    import hashlib
    import os
    import tarfile
    import time

    import requests
    from huggingface_hub import hf_hub_url

    wanted = {n.split("/")[-1] for n in names}
    url = hf_hub_url(REPO, ARCHIVE, repo_type="dataset", revision=REVISION)
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
    out = Path("/clips") / tag / "soma_uniform"
    out.mkdir(parents=True, exist_ok=True)
    found: dict[str, dict[str, object]] = {}
    started = time.time()
    members = 0
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        response.raw.decode_content = False
        with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
            for member in archive:
                members += 1
                base = member.name.split("/")[-1]
                if not member.isfile() or base not in wanted:
                    continue
                handle = archive.extractfile(member)
                assert handle is not None
                data = handle.read()
                target = out / member.name.lstrip("./")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                found[base] = {
                    "member": member.name,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
                if len(found) == len(wanted):
                    break
    receipt = {
        "repo": REPO,
        "revision": REVISION,
        "archive": ARCHIVE,
        "requested": sorted(wanted),
        "found": found,
        "missing": sorted(wanted - set(found)),
        "members_scanned": members,
        "seconds": round(time.time() - started, 1),
        "stopped_early": len(found) == len(wanted),
    }
    (Path("/clips") / tag / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    volume.commit()
    return {k: receipt[k] for k in ("missing", "members_scanned", "seconds", "stopped_early")} | {
        "found": len(found)
    }


@app.local_entrypoint()
def main(selection: str, tag: str = "demo-pool-v1") -> None:
    rows = [json.loads(line) for line in Path(selection).read_text().splitlines() if line.strip()]
    names = [row["move_soma_uniform_path"] for row in rows]
    print(json.dumps(extract.remote(names, tag)))
