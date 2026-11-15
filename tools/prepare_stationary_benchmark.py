#!/usr/bin/env python3
"""Prepare a portable fixed-base PiPER-pair nominal URR benchmark through the owner.

The fixture placement and actuator profile are explicit simulation assumptions.
Gripper aperture and grasp-centre facts come unchanged from canonical PiPER parts.
No hardware motion, paid job, training admission, or measured-fixture claim occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path, PurePosixPath
import shutil

import mujoco
import numpy as np
from sx_contracts.identity import canonical_json, file_digest
from sx_embodiments import PlacedEmbodiment, compose_embodiments
from sx_embodiments.embodiment import embodiment_from_definition
from sx_embodiments.kinematics import Kinematics
from sx_embodiments.known.piper import PIPER_SPEC
from sx_embodiments.layout import Bounds
from sx_embodiments.materialize import closure, materialize
from sx_embodiments.nominal_mujoco import (
    CollisionMode, NominalJoint, NominalProfile, RootMode, export_nominal_mujoco, mimic_projection,
)
from sx_embodiments.parts import ChainSpec, GripperSpec


def stationary_body():
    piper = embodiment_from_definition(PIPER_SPEC)
    return compose_embodiments('stationary-piper-pair', {
        'left': PlacedEmbodiment(piper, xyz=(0., .35, 0.)),
        'right': PlacedEmbodiment(piper, xyz=(0., -.35, 0.)),
    }, label='Nominal stationary PiPER pair')


def prepare_stationary_benchmark(output: Path):
    output = Path(output).resolve()
    if output.exists() or 'raw' in output.parts:
        raise ValueError('Preparation requires a new derived directory outside raw')
    body = stationary_body()
    components = {str(component.component_id): component for component in body.components}
    joints = []
    apertures = {}
    for name, coordinate in zip(body.joint_names, body.state.coordinates, strict=True):
        component = components[str(coordinate.instance)]
        part = component.part
        if isinstance(part, GripperSpec):
            if len(part.layout.axes) != 1 or part.travel_m is None:
                raise ValueError('Benchmark requires an owner-declared single-drive aperture relation')
            aperture = sum(part.travel_m)/2
            nominal = part.drive_from_aperture(aperture)
            if not np.isclose(part.aperture_from_drive(nominal), aperture):
                raise ValueError('Owner aperture relation failed round trip')
            kp, kd, effort, armature, scale = 600., 8., 10., .001, .005
        elif isinstance(part, ChainSpec):
            posture = part.ready if part.ready is not None else part.home
            if posture is None:
                raise ValueError('Canonical chain must declare its initial posture')
            nominal = posture[tuple(axis.name for axis in part.layout.axes).index(coordinate.axis.name)]
            kp, kd, effort, armature, scale = 120., 12., 30., .02, .15
        else:
            raise ValueError('Unaccounted native component')
        if not isinstance(coordinate.axis.bounds, Bounds) or not coordinate.axis.bounds.lower <= nominal <= coordinate.axis.bounds.upper:
            raise ValueError('Nominal state is outside canonical bounds')
        joints.append(NominalJoint(name,kp,kd,effort,armature,nominal,scale))
    roles = {f'{side}_hand': f'{side}/hand' for side in ('left','right')}
    for role, site in roles.items():
        component_id = str(body.site(site).component_id)
        names = [name for name,coordinate in zip(body.joint_names,body.state.coordinates,strict=True)
                 if str(coordinate.instance) == component_id]
        if len(names) != 1 or not isinstance(components[component_id].part, GripperSpec):
            raise ValueError('Each hand must bind its one owner gripper drive')
        apertures[role] = {'component_id': component_id, 'drive_joint': names[0]}
    kin = Kinematics.of(body.urdf_bytes)
    profile = NominalProfile(str(body.id),tuple(joints),(),tuple(sorted(kin.links)),(),
        (0.,0.,0.),(1.,0.,0.,0.),.002,(0.,0.,-9.81),(1.,.005,.0001),(
            'Nominal fixed fixture: canonical PiPER instances at Y +0.35/-0.35 m, facing +X; placement is not a measured installation.',
            'PD gains, force limits, armatures and action scales are explicit simulation parameters, not physical drive calibration.',
            'Native arm Ready posture and gripper aperture law/grasp centre are from the canonical owner parts.',
            'Ground-only collision approximation: no inter-arm or self-collision qualification, objects or grasp-success claim.',
            'URDF mimic relations are exported as simulator equalities; no invented finger coupling.',
        ),CollisionMode.GROUND_ONLY,RootMode.FIXED)
    output.mkdir(parents=True, mode=0o700)
    receipt = export_nominal_mujoco(body,profile,output/'robot')
    model = mujoco.MjModel.from_xml_path(str(output/'robot/scene.xml'))
    if np.any(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) or model.nu != len(body.joint_names):
        raise ValueError('Owner projection must have a fixed base and exactly one control per native coordinate')
    dependents = [{'joint':name,'leader':leader,'multiplier':multiplier,'offset':offset}
                  for name,(leader,multiplier,offset) in mimic_projection(kin).items()]
    assets = materialize(body).resolve()
    asset_files = []
    for relative,content in sorted(closure(body).items()):
        logical = PurePosixPath(relative)
        if logical.is_absolute() or '..' in logical.parts:
            raise ValueError('Nonrelative owner asset')
        source = (assets/relative).resolve()
        if not source.is_relative_to(assets):
            raise ValueError('Owner asset escaped its closure')
        dest = output/'assets'/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest)
        if str(file_digest(dest)) != str(content.sha256) or dest.stat().st_size != content.size_bytes:
            raise ValueError('Portable asset copy failed verification')
        asset_files.append({'path':str(dest.relative_to(output)),'sha256':str(content.sha256),'size_bytes':content.size_bytes})
    (output/'embodiment.json').write_text(body.to_json())
    (output/'nominal-profile.json').write_text(canonical_json(asdict(profile)))
    manifest = {'schema':'xmimic.stationary-benchmark/v1','embodiment_id':str(body.id),
        'urdf_sha256':receipt.urdf_sha256,'scene_sha256':receipt.scene_sha256,
        'owner_receipt_sha256':str(file_digest(output/'robot/receipt.json')),
        'joint_names':list(body.joint_names),'joint_units':[coordinate.axis.unit.value for coordinate in body.state.coordinates],
        'nominal_q':[joint.default_position for joint in joints],'action_scale':[joint.action_scale for joint in joints],
        'control_dt':.02,'simulation_substeps':10,'roles':roles,'apertures':apertures,
        'dependent_joints':dependents,'held_coordinates':[],
        'scope':'Nominal fixed-base physics only; no physical qualification or training admission',
        'aperture_authority':'Canonical GripperSpec.aperture_from_drive; meters, not normalized finger command',
        'assets':asset_files}
    (output/'benchmark.json').write_text(canonical_json(manifest))
    return body, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    body, manifest = prepare_stationary_benchmark(args.output)
    print(canonical_json({'embodiment_id':str(body.id),'output':str(args.output.resolve()),
                          'native_coordinates':len(body.joint_names),'roles':manifest['roles']}))
