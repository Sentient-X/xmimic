"""Derive XMimic metadata from the embodiment owner's nominal MuJoCo export.

The owner supplies geometry, native joint ordering, and explicit nominal PD values.
This adapter supplies no robot constants, hardware qualification or guessed symmetry.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from sx_contracts.identity import content_digest

from .model import load_model


@dataclass(frozen=True)
class TrainingProfile:
    """Explicit simulation randomization and capacity; lengths in metres, angles in radians.

    Reset vector entries are symmetric half-widths, matching Sim.reset_masked.
    A mirror is optional; without one configure wm_mirror=0.
    """
    com_body: str
    com_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    friction_range: tuple[float, float]
    joint_offset: float
    reset_pos: tuple[float, float, float]
    reset_rot: tuple[float, float, float]
    reset_lin_vel: tuple[float, float, float]
    reset_ang_vel: tuple[float, float, float]
    reset_joint_pos: float
    reset_joint_vel: float
    njmax: int
    nconmax: int
    mirror_permutation: tuple[int, ...] | None = None
    mirror_signs: tuple[int, ...] | None = None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_nominal_receipt(directory: Path) -> dict:
    """Verify owner profile and portable asset closure even when metadata exists."""
    directory = Path(directory)
    receipt = json.loads((directory / 'receipt.json').read_text())
    if receipt['scene_sha256'] != _sha(directory / 'scene.xml'):
        raise ValueError('scene does not match its owner export receipt')
    if not receipt['embodiment_id']:
        raise ValueError('owner receipt needs embodiment identity')
    nominal = receipt['profile']
    if str(content_digest(nominal)) != receipt['profile_sha256']:
        raise ValueError('nominal profile digest differs from receipt')
    if nominal['expected_embodiment_id'] != receipt['embodiment_id']:
        raise ValueError('nominal profile names another embodiment')
    files = receipt['files']
    if len({p for p, _ in files}) != len(files) or 'scene.xml' not in dict(files):
        raise ValueError('owner asset closure must contain scene exactly once')
    for relative, digest in files:
        candidate = directory / relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or not candidate.resolve().is_relative_to(directory.resolve()):
            raise ValueError('owner asset path escapes bundle')
        if _sha(candidate) != digest:
            raise ValueError('owner asset digest mismatch')
    return receipt


def adapt_nominal_bundle(directory: Path, profile: TrainingProfile) -> dict:
    """Write meta.json beside an owner export, validate through load_model, and return it.

    The scene and owner receipt remain unchanged. Failed validation removes only this
    newly written metadata; an existing meta.json is never overwritten.
    """
    directory = Path(directory)
    if 'raw' in directory.resolve().parts:
        raise ValueError('derive training metadata outside raw source custody')
    destination = directory / 'meta.json'
    if destination.exists():
        raise FileExistsError(destination)
    receipt = verify_nominal_receipt(directory)
    nominal = receipt['profile']
    names = receipt['joint_names']
    joints = nominal['joints']
    if not names or len(set(names)) != len(names) or [j['name'] for j in joints] != names:
        raise ValueError('owner profile must declare each native joint in order')
    model = mujoco.MjModel.from_xml_path(str(directory / 'scene.xml'))
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1:
        raise ValueError('exactly one floating root required')
    root = int(model.jnt_bodyid[free[0]])
    body_ids = [b for b in range(1, model.nbody) if model.body_rootid[b] == root]
    body_names = [model.body(b).name for b in body_ids]
    if any(not name for name in body_names) or len(set(body_names)) != len(body_names):
        raise ValueError('every robot link must have a distinct name')
    body_map = dict(zip(body_names, body_ids, strict=True))
    tracked = nominal['tracked_links']
    contacts = nominal['contacts']
    if not tracked or len(set(tracked)) != len(tracked) or not set(tracked) <= body_map.keys():
        raise ValueError('tracked links must be distinct robot links')
    if profile.com_body not in body_map:
        raise ValueError('COM randomization must target a robot link')
    com = np.asarray(profile.com_range)
    fr = np.asarray(profile.friction_range)
    if com.shape != (3, 2) or not np.isfinite(com).all() or np.any(com[:, 0] > com[:, 1]):
        raise ValueError('com_range must contain three finite ordered intervals')
    # XiLayout currently divides by the friction half-width, so a degenerate interval is unsafe.
    if fr.shape != (2,) or not np.isfinite(fr).all() or not 0 <= fr[0] < fr[1]:
        raise ValueError('friction_range must be positive-width and nonnegative')
    for field in ('reset_pos', 'reset_rot', 'reset_lin_vel', 'reset_ang_vel'):
        value = np.asarray(getattr(profile, field))
        if value.shape != (3,) or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f'{field} must contain three nonnegative finite half-widths')
    for field in ('joint_offset', 'reset_joint_pos', 'reset_joint_vel'):
        value = getattr(profile, field)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{field} must be finite and nonnegative')
    for field in ('njmax', 'nconmax'):
        if type(getattr(profile, field)) is not int or getattr(profile, field) <= 0:
            raise ValueError(f'{field} must be a positive integer')
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    if min(joint_ids) < 0:
        raise ValueError('native joint absent from model')
    if any(not model.jnt_limited[j] for j in joint_ids):
        raise ValueError('training requires explicit finite joint limits')
    if not np.allclose(model.qpos0[model.jnt_qposadr[joint_ids]], 0, atol=1e-12, rtol=0):
        raise ValueError('joint reference offsets require an explicit kinematic conversion')
    act = []
    for j in joint_ids:
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == j)
        if len(matches) != 1:
            raise ValueError('each native joint requires one actuator')
        act.append(int(matches[0]))
    index = {b: i for i, b in enumerate(body_ids)}
    native = {j: i for i, j in enumerate(joint_ids)}
    kin_joints, axes, matrices = [], [], []
    for b in body_ids:
        js = list(range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]))
        if b != root and (len(js) > 1 or any(j not in native for j in js)):
            raise ValueError('unrepresented moving coordinate in robot tree')
        j = js[0] if js and b != root else None
        kin_joints.append(native[j] if j is not None else -1)
        axes.append(model.jnt_axis[j].tolist() if j is not None else [0., 0., 0.])
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, model.body_quat[b])
        matrices.append(matrix.reshape(3, 3).tolist())
    ground = [g for g in range(model.ngeom) if model.geom_bodyid[g] == 0
              and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    if not ground:
        raise ValueError('training scene must include a world ground plane')
    sim_keys = ('timestep', 'integrator', 'cone', 'solver', 'iterations', 'tolerance',
                'ls_iterations', 'ls_tolerance', 'ccd_iterations', 'impratio')
    meta = {
        'embodiment_id': receipt['embodiment_id'], 'joint_names': names,
        'owner_receipt_sha256': _sha(directory / 'receipt.json'),
        'owner_receipt': receipt, 'training_profile': asdict(profile),
        'training_profile_sha256': str(content_digest(asdict(profile))),
        'qpos_adr': [int(model.jnt_qposadr[j]) for j in joint_ids],
        'dof_adr': [int(model.jnt_dofadr[j]) for j in joint_ids], 'actuator_of_joint': act,
        'free_qpos_adr': int(model.jnt_qposadr[free[0]]), 'free_dof_adr': int(model.jnt_dofadr[free[0]]),
        'default_joint_pos': [j['default_position'] for j in joints],
        'action_scale': [j['action_scale'] for j in joints],
        'q_lo': model.jnt_range[joint_ids, 0].tolist(), 'q_hi': model.jnt_range[joint_ids, 1].tolist(),
        'tracked_bodies': tracked, 'tracked_body_ids': [body_map[n] for n in tracked],
        'contact_bodies': [c['link'] for c in contacts], 'contact_heights': [c['height_m'] for c in contacts],
        'com_body': profile.com_body, 'com_body_id': body_map[profile.com_body], 'com_range': com.tolist(),
        'robot_geom_ids': [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in index],
        'ground_geom_ids': ground, 'blink_ground_geom_ids': ground,
        'ground_friction': model.geom_friction[ground].tolist(),
        'kin_body_names': body_names,
        'kin_parent': [-1 if b == root else index[int(model.body_parentid[b])] for b in body_ids],
        'kin_joint_idx': kin_joints, 'kin_axes': axes, 'kin_off_p': model.body_pos[body_ids].tolist(),
        'kin_off_R': matrices, 'kin_tracked_idx': [index[body_map[n]] for n in tracked],
        'dr': {field: getattr(profile, field) for field in ('friction_range', 'reset_pos', 'reset_rot',
              'reset_lin_vel', 'reset_ang_vel', 'reset_joint_pos', 'reset_joint_vel')},
        'sim': {**{k: getattr(model.opt, k) for k in sim_keys}, 'njmax': profile.njmax, 'nconmax': profile.nconmax},
        'counts': {k: getattr(model, k) for k in ('nq', 'nv', 'nu', 'nbody', 'ngeom')},
        'mujoco_version': mujoco.__version__, 'exact': {},
    }
    meta['dr']['joint_offset'] = {'.*': profile.joint_offset}
    if (profile.mirror_permutation is None) != (profile.mirror_signs is None):
        raise ValueError('mirror permutation and signs must be supplied together')
    if profile.mirror_permutation is not None:
        from .algorithm import Mirror
        from .sim import XiLayout
        Mirror(profile.mirror_permutation, profile.mirror_signs, XiLayout(meta))
        meta['mirror_permutation'] = list(profile.mirror_permutation)
        meta['mirror_signs'] = list(profile.mirror_signs)
    default = np.asarray(meta['default_joint_pos'])
    if np.any(default < meta['q_lo']) or np.any(default > meta['q_hi']):
        raise ValueError('default position outside native joint limits')
    encoded = json.dumps(meta, indent=2, allow_nan=False)
    with destination.open('x') as stream:
        stream.write(encoded)
    try:
        load_model(directory, motion_format='retargeted_npz')
    except Exception:
        destination.unlink()
        raise
    return meta
