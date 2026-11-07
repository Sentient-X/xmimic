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

from xmimic.artifact_cache import Provenance
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
    parser.add_argument(
        "--inputs", type=Path, required=True, help="prepare_embodiment_job output"
    )
    parser.add_argument(
        "--source-entry",
        required=True,
        help="Catalog source or pack that supplied motion",
    )
    parser.add_argument("--embodiment-id", required=True)
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


async def run(args):
    from sx import Sx
    from sx.credentials import api_key_for
    from sx._transport import ConnectionProfile, Endpoints
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
    cfg = Config(**raw)
    if cfg.iterations % cfg.log_every or cfg.ckpt_every % cfg.log_every:
        raise ValueError(
            "Managed iterations and checkpoint cadence must be multiples of log_every"
        )
    key = os.environ.get("SX_API_KEY") or api_key_for(ConnectionProfile.PRODUCTION)
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
        1,
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
    url = Endpoints.resolve(ConnectionProfile.PRODUCTION).for_door(CATALOG.name)
    with CatalogClient(url, api_key=key, project_id=ProjectId(args.project)) as catalog:
        client = Sx(project=args.project, api_key=key)
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
