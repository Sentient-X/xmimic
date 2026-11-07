"""The Unitree G1 tracking scene as a MuJoCo model, with the simulation options BlinkMimic applies at runtime.

``assets/g1/scene.xml`` and ``meta.json`` were exported from BlinkMimic's mjlab scene (tools/export_g1_from_blinkmimic.py);
the meshes are downloaded by tools/download_data.sh. The XML prints floats with about 7 significant digits, so the fields
it rounds (actuator gains, control ranges, armature, masses, inertias) are restored from the exact values in meta.json:
every compared model array then matches mjlab's.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "g1"


def load_meta(path: Path = ASSETS) -> dict:
    """Robot metadata: joint and body order, actuators, default pose, action scale, domain randomization, solver options."""
    meta = json.loads((Path(path) / "meta.json").read_text())
    feet = meta["contact_bodies"]
    if not feet or len(set(feet)) != len(feet) or not set(feet) <= set(meta["tracked_bodies"]):
        raise ValueError("contact_bodies must name distinct tracked bodies")
    heights = np.asarray(meta["contact_heights"])
    if heights.shape != (len(feet),) or not np.isfinite(heights).all() or np.any(heights < 0):
        raise ValueError("contact_heights must give a finite nonnegative height for each contact body")
    return meta


def contact_indices(meta: dict) -> list[int]:
    """Any number of explicitly declared support contacts, in tracked-body order."""
    return [meta["tracked_bodies"].index(name) for name in meta["contact_bodies"]]


def load_model(path: Path = ASSETS, *, motion_format: str | None = None) -> tuple[mujoco.MjModel, dict]:
    """(MjModel with BlinkMimic's solver options and exact parameter values, metadata)."""
    path = Path(path)
    if motion_format == "lafan1_csv" and path.resolve() != ASSETS.resolve():
        raise ValueError("LAFAN1 CSVs are G1-specific; retarget motions to the selected embodiment first")
    meta = load_meta(path)
    if path.resolve() != ASSETS.resolve() and not meta.get("embodiment_id"):
        raise ValueError("a custom robot bundle must bind its canonical embodiment_id")
    m = mujoco.MjModel.from_xml_path(str(path / "scene.xml"))
    s = meta["sim"]
    m.opt.timestep = s["timestep"]
    m.opt.integrator = s["integrator"]
    m.opt.cone = s["cone"]
    m.opt.solver = s["solver"]
    m.opt.iterations = s["iterations"]
    m.opt.tolerance = s["tolerance"]
    m.opt.ls_iterations = s["ls_iterations"]
    m.opt.ls_tolerance = s["ls_tolerance"]
    m.opt.ccd_iterations = s["ccd_iterations"]
    m.opt.impratio = s["impratio"]
    for field, value in meta["exact"].items():
        getattr(m, field)[:] = np.asarray(value, dtype=np.float64).reshape(getattr(m, field).shape)
    if not np.isclose(m.opt.timestep, 0.005, rtol=0, atol=1e-12):
        raise ValueError("simulation must use 5 ms steps for the trainer's 50 Hz control rate")
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if any(type(meta[key]) is not int for key in ("free_qpos_adr", "free_dof_adr")):
        raise ValueError("floating-root addresses must be integers")
    if len(free) != 1 or (meta["free_qpos_adr"], meta["free_dof_adr"]) != (m.jnt_qposadr[free[0]], m.jnt_dofadr[free[0]]):
        raise ValueError("the robot bundle must identify its single floating root")
    names = meta["joint_names"]
    if len(set(names)) != len(names) or not names:
        raise ValueError("joint_names must be nonempty and unique")
    if len(names) != m.njnt - 1 or len(names) != m.nu or m.na != 0:
        raise ValueError("every moving coordinate and stateless actuator must be represented")
    for key in ("qpos_adr", "dof_adr", "actuator_of_joint", "default_joint_pos", "action_scale", "q_lo", "q_hi"):
        values = np.asarray(meta[key])
        if values.shape != (len(names),) or not np.isfinite(values).all():
            raise ValueError(f"{key} must contain one finite value per joint")
    if np.any(np.asarray(meta["action_scale"]) <= 0):
        raise ValueError("action_scale must be positive")
    for key in ("qpos_adr", "dof_adr", "actuator_of_joint"):
        if any(type(value) is not int for value in meta[key]):
            raise ValueError(f"{key} must contain integer indices")
    for name, qa, da, actuator in zip(names, meta["qpos_adr"], meta["dof_adr"], meta["actuator_of_joint"], strict=True):
        matches = [j for j in range(m.njnt) if m.joint(j).name in (name, "robot/" + name)]
        if len(matches) != 1:
            raise ValueError(f"joint {name!r} must resolve once in the simulation")
        j = matches[0]
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"joint {name!r}: only scalar revolute coordinates are supported")
        if not np.allclose(m.jnt_pos[j], 0, rtol=0, atol=1e-12) or m.body_jntnum[m.jnt_bodyid[j]] != 1:
            raise ValueError(f"joint {name!r}: kinematics requires one hinge at its link origin")
        if qa != m.jnt_qposadr[j] or da != m.jnt_dofadr[j]:
            raise ValueError(f"joint {name!r}: state addresses disagree with the simulation")
        if not isinstance(actuator, int) or not 0 <= actuator < m.nu or m.actuator_trnid[actuator, 0] != j:
            raise ValueError(f"joint {name!r}: actuator does not drive this coordinate")
        if m.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT or not np.allclose(m.actuator_gear[actuator], [1, 0, 0, 0, 0, 0]):
            raise ValueError(f"joint {name!r}: requires a direct joint position actuator")
        gain, bias = m.actuator_gainprm[actuator], m.actuator_biasprm[actuator]
        if m.actuator_gaintype[actuator] != mujoco.mjtGain.mjGAIN_FIXED or m.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_AFFINE or gain[0] <= 0 or not np.isclose(bias[1], -gain[0]):
            raise ValueError(f"joint {name!r}: control must be a position target")
    return m, meta
