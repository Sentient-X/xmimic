#!/usr/bin/env python3
"""Run a prepared embodiment job on managed Modal CPU, cache it, then train on H100."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import fields
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xmimic.artifact_cache import Provenance, _atomic, _json, content_hash
from xmimic.config import Config
from xmimic.managed_pipeline import (
    JobSettings,
    Stage,
    completed_output,
    publish_inputs,
    submit_stage,
    training_candidate,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--inputs", type=Path, help="Existing prepared input closure")
    target.add_argument("--embodiment", help="Registered sx-embodiments name")
    target.add_argument(
        "--embodiment-document", type=Path, help="Pinned canonical body document"
    )
    parser.add_argument("--source", type=Path, help="Complete human SOMA BVH clip")
    parser.add_argument("--nominal-profile", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--training-profile", type=Path)
    parser.add_argument(
        "--profile", choices=("production", "preview"), default="production"
    )
    parser.add_argument(
        "--source-entry",
        required=True,
        help="Catalog source or pack that supplied motion",
    )
    parser.add_argument("--embodiment-id", help="Optional expected exact body digest")
    parser.add_argument("--clip-id", required=True)
    parser.add_argument(
        "--source-group", required=True, help="original/mirror family identity"
    )
    parser.add_argument("--shape-sha256")
    parser.add_argument("--cpu-image", required=True, help="built image@sha256:digest")
    parser.add_argument("--gpu-image", required=True, help="built image@sha256:digest")
    parser.add_argument("--revision", required=True, help="exact XMimic source commit")
    parser.add_argument("--license-id", required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="persistent journal; reuse to reattach",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path.home() / ".cache/sentientx/xmimic"
    )
    parser.add_argument("--cpu-timeout", type=int, default=600)
    parser.add_argument("--gpu-timeout", type=int, default=600)
    parser.add_argument("--cpu-budget-micros", type=int, default=1000000)
    parser.add_argument("--gpu-budget-micros", type=int, default=5000000)
    return parser.parse_args()


def prepare_inputs(args):
    """Resolve a selected body once and retain its exact portable input closure."""
    from sx_contracts.identity import file_digest
    from sx_embodiments import Embodiment, embodiments
    from tools.prepare_embodiment_job import prepare
    from xmimic.retarget_source import ParseLimits, parse_soma_bvh

    if args.inputs is None:
        paths = (args.source, args.nominal_profile, args.recipe, args.training_profile)
        if any(path is None for path in paths):
            raise ValueError(
                "Embodiment selection requires source, nominal-profile, recipe and training-profile"
            )
        if args.source.stat().st_size > ParseLimits().max_bytes:
            raise ValueError("Source exceeds the parser size limit")
        body = (
            Embodiment.from_json(args.embodiment_document.read_text())
            if args.embodiment_document is not None
            else embodiments[args.embodiment]
        )
        import hashlib

        payloads = [path.read_bytes() for path in paths]
        identity = {
            "embodiment_id": str(body.id),
            "source_entry": args.source_entry,
            "files": [hashlib.sha256(payload).hexdigest() for payload in payloads],
        }
        key = content_hash(identity)
        directory = args.run_dir / "inputs" / key
        args.run_dir.mkdir(parents=True, exist_ok=True)
        request = args.run_dir / "preparation-request.json"
        state = (
            json.loads(request.read_bytes())
            if request.exists()
            else {"selection": identity}
        )
        if state.get("selection") != identity:
            raise ValueError(
                "Run directory already binds another body or source; choose another run directory"
            )
        _atomic(request, _json(state))
        if not directory.exists():
            if "closure" in state:
                raise ValueError(
                    "Retained preparation is missing; reconcile the existing run"
                )
            prepare(body, *payloads, source_entry=args.source_entry, output=directory)
            state["closure"] = {
                path.relative_to(directory).as_posix(): str(file_digest(path))
                for path in sorted(directory.rglob("*"))
                if path.is_file()
            }
            _atomic(request, _json(state))
        if "closure" not in state:
            raise ValueError(
                "Preparation has no acknowledged content closure; reconcile before submission"
            )
        members = sorted(directory.rglob("*"))
        if directory.is_symlink() or any(path.is_symlink() for path in members):
            raise ValueError("Retained preparation contains a linked member")
        actual = {
            path.relative_to(directory).as_posix(): str(file_digest(path))
            for path in members
            if path.is_file()
        }
        if actual != state["closure"]:
            raise ValueError(
                "Retained preparation bytes differ from the selected input closure"
            )
        if (
            Embodiment.from_json((directory / "embodiment.json").read_text()).id
            != body.id
        ):
            raise ValueError("Retained preparation body differs from selection")
        args.inputs = directory
    elif any(
        path is not None
        for path in (
            args.source,
            args.nominal_profile,
            args.recipe,
            args.training_profile,
        )
    ):
        raise ValueError("Prepared inputs already own source and profiles")
    body = Embodiment.from_json((args.inputs / "embodiment.json").read_text())
    if args.embodiment_id is not None and args.embodiment_id != str(body.id):
        raise ValueError("Selected embodiment differs from the expected identity")
    args.embodiment_id = str(body.id)
    source = args.inputs / "source.bvh"
    if source.stat().st_size > ParseLimits().max_bytes:
        raise ValueError("Source exceeds the parser size limit")
    from xmimic.retarget_workflow import recipe_from_json

    recipe = recipe_from_json((args.inputs / "recipe.json").read_bytes())
    return parse_soma_bvh(
        source.read_bytes(), world_from_source=recipe.world_from_source
    ).frame_count


async def run(args):
    from sx.client import Sx
    from sx.credentials import api_key_for, environment_endpoints
    from sx._transport import ConnectionProfile
    from sx_auth.principal import ProjectId
    from sx_catalog_client import CatalogClient
    from sx_service.registry import CATALOG

    raw = json.loads(args.training_config.read_bytes())
    forbidden = {"robot_dir", "motion_dir", "log_dir", "motion_format", "layout"}
    if (
        not isinstance(raw, dict)
        or not set(raw) <= {f.name for f in fields(Config)} - forbidden
    ):
        raise ValueError("Training config contains unknown or managed-path options")
    defaults = Config()
    if any(
        type(value) is not type(getattr(defaults, name)) for name, value in raw.items()
    ):
        raise ValueError("Training option type differs from Config")
    cfg = Config(**raw)
    if cfg.iterations % cfg.log_every or cfg.ckpt_every % cfg.log_every:
        raise ValueError(
            "Managed iterations and checkpoint cadence must be multiples of log_every"
        )
    frames = prepare_inputs(args)
    connection = ConnectionProfile(args.profile)
    key = os.environ.get("SX_API_KEY") or api_key_for(connection)
    if not key:
        raise ValueError("Sign in to SentientX before launching this project workflow")
    provenance = Provenance(
        "https://github.com/Sentient-X/xmimic",
        args.revision,
        args.license_id,
        "xmimic.managed_pipeline",
    )
    cpu = JobSettings(
        args.cpu_image,
        args.cpu_timeout,
        args.cpu_budget_micros,
        4,
        16,
        frames,
        {
            "embodiment_id": args.embodiment_id,
            "clip_id": args.clip_id,
            "source_group": args.source_group,
            "shape_sha256": args.shape_sha256,
        },
    )
    # Validate the GPU allocation before buying CPU work. Input binding arrives
    # from the retained retarget receipt, never from an inferred array width.
    gpu = JobSettings(
        args.gpu_image,
        args.gpu_timeout,
        args.gpu_budget_micros,
        8,
        32,
        cfg.iterations,
        {},
    )
    endpoints = environment_endpoints(connection)
    url = endpoints.for_door(CATALOG.name)
    with CatalogClient(url, api_key=key, project_id=ProjectId(args.project)) as catalog:
        client = Sx(
            project=args.project, api_key=key, profile=connection, endpoints=endpoints
        )
        try:
            inputs = publish_inputs(
                catalog,
                project=args.project,
                directory=args.inputs,
                journal=args.run_dir / "publications",
                provenance=provenance,
                parent_entry=args.source_entry,
                training_eligible=True,
            )
            operation = await submit_stage(
                client, inputs, Stage.RETARGET, cpu, args.run_dir
            )
            print("CPU operation:", operation.ref, flush=True)
            output = await completed_output(operation, catalog, args.cache_dir)
            candidate = training_candidate(output)
            gpu_inputs = publish_inputs(
                catalog,
                project=args.project,
                directory=output.directory,
                journal=args.run_dir / "publications",
                provenance=provenance,
                parent_entry=output.entry,
                manifest=candidate,
            )
            receipt = json.loads((output.directory / "candidate.json").read_bytes())
            gpu.config.update(
                training=raw,
                source_binding=receipt["cache_manifest_sha256"],
                embodiment_id=args.embodiment_id,
            )
            operation = await submit_stage(
                client, gpu_inputs, Stage.TRAIN, gpu, args.run_dir
            )
            print("GPU operation:", operation.ref, flush=True)
            result = await completed_output(operation, catalog, args.cache_dir)
            print(
                json.dumps(
                    {
                        "entry": result.entry,
                        "manifest_sha256": result.sha256,
                        "local_output": str(result.directory),
                    },
                    indent=2,
                )
            )
        finally:
            await client.close()


if __name__ == "__main__":
    asyncio.run(run(arguments()))
