#!/usr/bin/env python3
"""Prepare complete immutable inputs for a managed CPU retarget Job; submit nothing."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from sx_contracts.identity import canonical_json, content_digest, file_digest
from sx_embodiments import Embodiment, embodiments
from sx_embodiments.materialize import closure, materialize
from sx_embodiments.nominal_mujoco import export_nominal_mujoco, nominal_profile_from_json

from xmimic.embodiment_bundle import TrainingProfile, adapt_nominal_bundle
from xmimic.retarget_source import ParseLimits, parse_soma_bvh
from xmimic.retarget_target import prepare_target
from xmimic.retarget_workflow import recipe_from_json


def training_profile(payload):
    def pairs(values):
        result = {}
        for key,value in values:
            if key in result:
                raise ValueError('Duplicate training profile key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('Nonfinite training profile value')
    values = json.loads(payload,object_pairs_hook=pairs,parse_constant=invalid)
    if not isinstance(values,dict) or set(values) != {f.name for f in fields(TrainingProfile)}:
        raise ValueError('Training profile fields differ from the supported artifact')
    return TrainingProfile(**values)


def prepare(body, source, nominal_bytes, recipe_bytes, training_bytes, *, source_entry, output):
    if not isinstance(source_entry,str) or not source_entry.strip():
        raise ValueError('Existing Catalog source entry lineage is required')
    output = Path(output).absolute()
    if output.exists() or 'raw' in output.resolve().parts:
        raise ValueError('Preparation requires a new derived output directory outside raw')
    nominal = nominal_profile_from_json(nominal_bytes)
    recipe = recipe_from_json(recipe_bytes)
    training = training_profile(training_bytes)
    target = prepare_target(body,recipe.target,expected_embodiment_id=nominal.expected_embodiment_id)
    if dict(nominal.passive_holds) != dict(recipe.target.nominal_passive):
        raise ValueError('Simulation and retarget passive projections differ')
    clip = parse_soma_bvh(source,world_from_source=recipe.world_from_source)
    clip.joint_index(recipe.source_root)
    for mapping in recipe.target.landmarks:
        clip.joint_index(mapping.source)
    for mapping in recipe.segment_targets:
        for parent, child, _ in mapping.segments:
            clip.joint_index(parent)
            clip.joint_index(child)
    initial = dict(recipe.initial_q)
    if set(initial) != set(body.joint_names) or any(
        not lo <= initial[name] <= hi for name,lo,hi in zip(body.joint_names,target.binding.lower,target.binding.upper,strict=True)
    ) or any(initial[name] != value for name,value in recipe.target.holds):
        raise ValueError('Initial retarget state does not account for exact bounded native coordinates')
    assets = materialize(body).resolve()
    asset_files = closure(body)
    output.parent.mkdir(parents=True,exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.prepare-',dir=output.parent))
    try:
        # Copy the complete owning closure, including the original descriptions;
        # no dependency on a laptop-global asset cache survives this boundary.
        for relative,content in asset_files.items():
            logical = PurePosixPath(relative)
            if logical.is_absolute() or '..' in logical.parts or str(logical) != relative:
                raise ValueError('Canonical asset closure contains a nonrelative path')
            original = (assets/relative).resolve()
            if not original.is_relative_to(assets):
                raise ValueError('Canonical asset escaped its verified closure')
            destination = staging/'assets'/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(original,destination)
            if str(file_digest(destination)) != str(content.sha256):
                raise ValueError('Canonical asset changed during preparation')
        receipt = export_nominal_mujoco(body,nominal,staging/'robot')
        adapt_nominal_bundle(staging/'robot',training)
        (staging/'embodiment.json').write_text(body.to_json())
        (staging/'source.bvh').write_bytes(source)
        (staging/'recipe.json').write_text(canonical_json(asdict(recipe)))
        (staging/'nominal-profile.json').write_text(canonical_json(asdict(nominal)))
        (staging/'training-profile.json').write_text(canonical_json(asdict(training)))
        report = {'embodiment_id':str(body.id),'urdf_sha256':receipt.urdf_sha256,
                  'source_entry':source_entry,'source_sha256':hashlib.sha256(source).hexdigest(),
                  'source_frame_time_literal':clip.frame_time_literal,'source_frames':clip.frame_count,
                  'target_binding_sha256':target.binding.sha256,'nominal_profile_sha256':nominal.sha256,
                  'recipe_sha256':str(content_digest(asdict(recipe))),
                  'training_profile_sha256':str(content_digest(asdict(training))),
                  'scope':'prepared input closure; retarget feasibility and training admission not established',
                  'training_admitted':False}
        (staging/'preparation.json').write_text(canonical_json(report))
        os.rename(staging,output)
        return report
    except BaseException:
        shutil.rmtree(staging)  # only this new derived staging directory
        raise


def _bounded_read(path, limit):
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError('Input file exceeds its supported preparation bound')
    return path.read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    body = parser.add_mutually_exclusive_group(required=True)
    body.add_argument('--embodiment')
    body.add_argument('--embodiment-document',type=Path)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--nominal-profile',type=Path,required=True)
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--training-profile',type=Path,required=True)
    parser.add_argument('--source-entry',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    embodiment = (Embodiment.from_json(_bounded_read(args.embodiment_document,8*1024**2).decode())
                  if args.embodiment_document is not None else embodiments[args.embodiment])
    report = prepare(embodiment,_bounded_read(args.source,ParseLimits().max_bytes),_bounded_read(args.nominal_profile,1024**2),
                     _bounded_read(args.recipe,1024**2),_bounded_read(args.training_profile,1024**2),
                     source_entry=args.source_entry,output=args.output)
    print(canonical_json(report))


if __name__ == '__main__':
    main()
