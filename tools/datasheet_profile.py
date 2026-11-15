"""Derive an experiment robot bundle whose actuators follow Damiao datasheet torque ratings instead of nominal role gains.

    python tools/datasheet_profile.py <robot_dir> <out_dir> [--ratings ratings.json] [--check]

Interim profile until sx-embodiments exports the real ratings: body geometry, inertials and embodiment_id are unchanged;
only actuator effort, armature, PD gains and action scale change, and a sagittal mirror is added when the geometry gives one. Gains follow the
BeyondMimic/mjlab rule the G1 baseline uses: kp = armature * w^2, kd = 2 * zeta * armature * w with w = 2 pi 10 Hz and
zeta = 2, action scale = 0.25 * effort / kp. Armatures are reflected-inertia estimates (rotor inertia x gear ratio^2,
order of magnitude; no Damiao rotor inertia is in the layer yet). The waist and ankle linkages need kp above the upper
body's gravity stiffness (about 45 Nm/rad): with a G1-class 0.0072 armature (kp 28) the torso folds forward under PD. ``--check`` runs a CPU PD hold of the default pose for 3 s and reports root
height and peak torque use.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np

W = 2 * math.pi * 10.0
ZETA = 2.0
# joint-name pattern -> (motor, peak torque Nm, estimated armature kg m^2, source)
RATINGS = [
    (r"_hip_(pitch|roll)_joint$", "DM10010", 150.0, 0.050, "embodiment ratings; armature estimated (not published)"),
    (r"_knee_joint$", "DM10010", 150.0, 0.050, "embodiment ratings; armature estimated (not published)"),
    (r"_hip_yaw_joint$", "DM8009", 40.0, 0.020, "embodiment ratings"),
    (r"^torso_joint$", "DM8009", 40.0, 0.020, "embodiment ratings"),
    # two DM4340 through a linkage at unit lever ratio: an upper bound, not measured
    (r"_foot[12]_joint$", "2xDM4340 linkage", 54.0, 0.064, "embodiment ratings (linkage upper bound)"),
    (r"^waist_(roll|pitch)_joint$", "2xDM4340 linkage", 54.0, 0.064, "embodiment ratings (linkage upper bound)"),
    (r"_shoulder_(pitch|roll)_joint$", "DM8009P", 40.0, 0.020, "embodiment ratings; joint mapping assumed"),
    (r"_(upper_arm_yaw|elbow_pitch)_joint$", "DM4340", 27.0, 0.032, "embodiment ratings; joint mapping assumed"),
    (r"_(forearm_yaw|wrist_pitch|wrist_roll)_joint$", "DM4310", 7.0, 0.004, "embodiment ratings; armature estimated"),
    (r"_finger_joint$", "YUBI gripper (unrated)", 5.0, 0.004, "nominal: no rating supplied"),
]
# The default pose stays the bundle's (zero): the policy acts as a residual on the reference (target = q_ref + scale * a),
# so the default only centres observations. A hand-set crouch is wrong here: the left and right knees have opposite axes.


# Stated assumptions for drives the layer publishes no armature for (not datasheet numbers): reflected rotor inertia,
# order of magnitude.
ASSUMED_ARMATURE = {"damiao_dm10010": 0.05, "feetech_sts3215": 0.004}


def layer_rating(entry: dict):
    """(drive, effort, armature, source) from one joint of a ratings file exported from sx_embodiments.joint_rating."""
    armature, source = entry["armature"], "sx-embodiments rating"
    if armature is None:
        armature, source = ASSUMED_ARMATURE[entry["drive"]], "sx-embodiments rating; armature assumed (not published)"
    return entry["drive"], float(entry["effort"]), float(armature), source


def rating(name: str):
    for pattern, motor, effort, armature, source in RATINGS:
        if re.search(pattern, name):
            return motor, effort, armature, source
    raise ValueError(f"no rating for {name}")


def geometric_mirror(scene: Path, joint_names: list[str]):
    """Signed joint permutation of the sagittal reflection. Partners pair by name (every left/right token swapped, so a
    left hand's left finger mirrors the right hand's right finger); each sign comes from the model's world joint axes at
    the zero pose, the partner's axis being the reflected pseudovector (-x, y, -z) up to sign. None unless every pair's
    axes agree to 0.99 and the result is an involution. Anchors are not compared: this body's linkage pivots are not
    placed symmetrically (up to 8 cm apart), although their axes are."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    axis = d.xaxis[ids]
    swap = {"left": "right", "right": "left"}
    perm, signs = [], []
    for j, name in enumerate(joint_names):
        partner = "_".join(swap.get(t, t) for t in name.split("_"))
        if partner not in joint_names:
            return None
        k = joint_names.index(partner)
        dot = float(axis[k] @ (axis[j] * [-1, 1, -1]))
        if abs(dot) < 0.99:
            return None
        perm.append(k)
        signs.append(1 if dot > 0 else -1)
    perm_a, sign_a = np.asarray(perm), np.asarray(signs)
    if not np.array_equal(perm_a[perm_a], np.arange(len(ids))) or not np.all(sign_a * sign_a[perm_a] == 1):
        return None
    return perm, signs


def derive(robot: Path, out: Path, ratings: dict | None = None) -> dict:
    meta = json.loads((robot / "meta.json").read_text())
    scene = (robot / "scene.xml").read_text()
    shutil.copytree(robot, out, ignore=shutil.ignore_patterns("scene.xml", "meta.json"), symlinks=True)
    table, scales, defaults = [], [], []
    for name in meta["joint_names"]:
        motor, effort, armature, source = layer_rating(ratings["joints"][name]) if ratings else rating(name)
        kp, kd = armature * W * W, 2 * ZETA * armature * W
        q0 = float(meta["default_joint_pos"][meta["joint_names"].index(name)])
        table.append({"joint": name, "motor": motor, "effort_nm": effort, "armature": armature, "kp": round(kp, 4),
                      "kd": round(kd, 4), "action_scale": round(0.25 * effort / kp, 4), "default": q0, "source": source,
                      "rated_velocity": ratings["joints"][name]["velocity"] if ratings else None})
        scales.append(0.25 * effort / kp)
        defaults.append(q0)
        joint = re.compile(rf'(<joint name="{re.escape(name)}"[^>]*?armature=")[^"]*(")')
        scene, n = joint.subn(rf"\g<1>{armature!r}\g<2>", scene)
        actuator = re.compile(rf'(<(?:position|general)[^>]*joint="{re.escape(name)}"[^>]*?/>)')
        match = actuator.search(scene)
        if n != 1 or match is None:
            raise ValueError(f"{name}: expected one joint and one actuator element")
        element = match.group(1)
        new = re.sub(r'forcerange="[^"]*"', f'forcerange="{-effort!r} {effort!r}"', element)
        if "<position" in element:
            new = re.sub(r'\bkp="[^"]*"', f'kp="{kp!r}"', new)
            new = re.sub(r'\bkv="[^"]*"', f'kv="{kd!r}"', new)
        else:
            new = re.sub(r'gainprm="[^"]*"', f'gainprm="{kp!r}"', new)
            new = re.sub(r'biasprm="[^"]*"', f'biasprm="0 {-kp!r} {-kd!r}"', new)
        scene = scene.replace(element, new, 1)
    meta["action_scale"], meta["default_joint_pos"] = scales, defaults
    mirror = geometric_mirror(robot / "scene.xml", meta["joint_names"])
    if mirror is not None and meta.get("mirror_permutation") is None:
        meta["mirror_permutation"], meta["mirror_signs"] = mirror
    meta["actuator_profile"] = {"name": "sx-embodiments-ratings" if ratings else "damiao-embodiment-ratings-v2",
                                "ratings_source": ratings["source"] if ratings else "table in tools/datasheet_profile.py", "rule": "kp=arm*w^2, kd=2*zeta*arm*w, w=2pi*10, zeta=2, "
                                "scale=0.25*effort/kp", "joints": table}
    (out / "scene.xml").write_text(scene)
    (out / "meta.json").write_text(json.dumps(meta))
    return meta


def check(out: Path, seconds: float = 3.0) -> dict:
    """Static torque capacity and the PD-hold behaviour of the default pose.

    Static: the joint torques that hold each pose with the weight split over both feet (tau = g(q) - sum J_foot^T F),
    against each actuator's rating, at the default pose and a deep squat. PD hold: constant targets with no policy. A
    PD-only stand needs ankle-pitch kp above m g h (about 240 Nm/rad here, the G1 baseline is 28.5): it is expected to
    topple slowly about the ankles and is reported, not required; the tracking policy balances actively."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
    meta = json.loads((out / "meta.json").read_text())
    names, qadr, dadr = meta["joint_names"], np.asarray(meta["qpos_adr"]), np.asarray(meta["dof_adr"])
    act = np.asarray(meta["actuator_of_joint"])
    lim = m.actuator_forcerange[act, 1]
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in meta["contact_bodies"]]
    weight = float(np.sum(m.body_mass) * 9.81)
    result = {"mass_kg": float(np.sum(m.body_mass)), "static": {}}
    q_def = np.asarray(meta["default_joint_pos"])
    squat = q_def.copy()
    # deep squat in this body's signs (the reference clips bend the right knee, hip and ankle pitch positive, the left
    # negative)
    for side, sign in (("right", 1.0), ("left", -1.0)):
        for joint, value in (("knee", 1.2), ("hip_pitch", 0.6), ("foot1", 0.6)):
            squat[names.index(f"{side}_{joint}_joint")] = sign * value
    for label, q0 in (("default", q_def), ("squat", squat)):
        d = mujoco.MjData(m)
        d.qpos[qadr] = q0
        mujoco.mj_forward(m, d)
        tau = d.qfrc_bias.copy()
        for b in feet:
            jac = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jac, None, b)
            tau -= jac.T @ np.array([0.0, 0.0, weight / len(feet)])
        frac = np.abs(tau[dadr]) / lim
        worst = np.argsort(-frac)[:3]
        result["static"][label] = {"peak_fraction": float(frac.max()),
                                   "worst": [[names[i], round(float(tau[dadr][i]), 2), float(lim[i])] for i in worst]}
    d = mujoco.MjData(m)
    d.qpos[qadr] = q_def
    mujoco.mj_forward(m, d)
    fq = meta["free_qpos_adr"]
    while d.ncon == 0 and d.qpos[fq + 2] > 0:
        d.qpos[fq + 2] -= 0.0005
        mujoco.mj_forward(m, d)
    h0, heights = float(d.qpos[fq + 2]), []
    for _ in range(int(seconds / m.opt.timestep)):
        d.ctrl[act] = q_def
        mujoco.mj_step(m, d)
        heights.append(float(d.qpos[fq + 2]))
    upright = next((i for i, h in enumerate(heights) if h < 0.8 * h0), None)
    result["pd_hold"] = {"root_height_start_m": h0, "root_height_end_m": heights[-1],
                         "upright_s": seconds if upright is None else upright * m.opt.timestep}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("robot", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ratings", type=Path, help="joint ratings exported from sx_embodiments.joint_rating (preferred)")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; derive into a new directory")
    derive(a.robot, a.out, json.loads(a.ratings.read_text()) if a.ratings else None)
    if a.check:
        result = check(a.out)
        (a.out / "pd-hold.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
