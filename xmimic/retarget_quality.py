"""Explicit nominal reference projection; never physical or self-collision qualification."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import mujoco
import numpy as np

from .embodiment_bundle import verify_nominal_receipt
from .model import load_model


@dataclass(frozen=True)
class NominalQuality:
    position_tolerance_m: float
    orientation_tolerance_rad: float
    support_links: tuple[str, ...]
    ground_clearance_m: float
    max_root_shift_m: float
    time_scale: float
    max_joint_speed_rad_s: float
    max_ground_penetration_m: float

    def __post_init__(self):
        object.__setattr__(self, "support_links", tuple(self.support_links))
        if (
            not self.support_links
            or len(set(self.support_links)) != len(self.support_links)
            or any(not isinstance(s, str) or not s for s in self.support_links)
        ):
            raise ValueError("Explicit distinct nominal support links required")
        for key, value in asdict(self).items():
            if key == "support_links":
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(
                    "Nominal quality values must be finite nonnegative numbers"
                )
        if (
            min(
                self.position_tolerance_m,
                self.orientation_tolerance_rad,
                self.max_root_shift_m,
                self.max_joint_speed_rad_s,
            )
            <= 0
            or self.time_scale < 1
        ):
            raise ValueError("Positive bounds and time dilation >=1 required")


def geometry_lowest_z(model, data, geom):
    """Exact vertical support of MuJoCo's mesh vertices or supported primitives."""
    rotation = data.geom_xmat[geom].reshape(3, 3)
    center = data.geom_xpos[geom, 2]
    kind = model.geom_type[geom]
    size = model.geom_size[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        return float(
            center + np.min(model.mesh_vert[start : start + count] @ rotation[2])
        )
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return float(center - np.abs(rotation[2]) @ size)
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(center - size[0])
    if kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        extent = size[1] * abs(rotation[2, 2])
        extent += size[0] * (
            1
            if kind == mujoco.mjtGeom.mjGEOM_CAPSULE
            else np.linalg.norm(rotation[2, :2])
        )
        return float(center - extent)
    raise ValueError("Unsupported nominal collision geometry for floor projection")


def qualify_motion(
    arrays,
    position_residual_m,
    orientation_residual_rad,
    robot_dir: Path,
    quality: NominalQuality,
):
    """Keep all q samples; translate whole poses vertically and dilate the whole clock.

    Targets share each frame's uniform translation, so relative landmark residuals
    are unchanged. The original clock, root trajectory and solver gate stay in the
    retained candidate. Contact labels, support balance and hardware limits are not
    inferred here. A successful result is a nominal simulation reference only.
    """
    receipt = verify_nominal_receipt(robot_dir)
    if receipt["profile"].get("collision_mode") != "ground_only":
        raise ValueError(
            "Nominal floor projection requires explicit ground_only simulation"
        )
    model, meta = load_model(robot_dir, motion_format="retargeted_npz")
    q = np.asarray(arrays["q"], dtype=float)
    root_p = np.array(arrays["root_p"], dtype=float, copy=True)
    root_q = np.asarray(arrays["root_quat"], dtype=float)
    fps = float(arrays["fps"])
    if (
        str(np.asarray(arrays["embodiment_id"]).item()) != meta["embodiment_id"]
        or np.asarray(arrays["joint_names"]).tolist() != meta["joint_names"]
    ):
        raise ValueError("Nominal projection body or native coordinate order differs")
    if (
        q.ndim != 2
        or len(q) < 2
        or q.shape[1] != len(meta["joint_names"])
        or root_p.shape != (len(q), 3)
        or root_q.shape != (len(q), 4)
        or not all(np.isfinite(v).all() for v in (q, root_p, root_q))
        or not math.isfinite(fps)
        or fps <= 0
        or not np.allclose(np.linalg.norm(root_q, axis=1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Malformed nominal motion arrays")
    pe, re = np.asarray(position_residual_m), np.asarray(orientation_residual_rad)
    if (
        pe.ndim != 2
        or pe.shape != re.shape
        or len(pe) != len(q)
        or not np.isfinite(pe).all()
        or not np.isfinite(re).all()
        or np.any(pe < 0)
        or np.any(re < 0)
    ):
        raise ValueError("Malformed retained residuals")
    floor = model.geom("nominal_floor").id
    if (
        model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE
        or np.any(model.geom_pos[floor] != 0)
        or not np.allclose(model.geom_quat[floor], [1, 0, 0, 0])
    ):
        raise ValueError(
            "Nominal projection requires the declared horizontal floor at z=0"
        )
    collision = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] != 0
        and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    if (model.geom_contype[floor], model.geom_conaffinity[floor]) != (1, 2) or any(
        (model.geom_contype[g], model.geom_conaffinity[g]) != (2, 1) for g in collision
    ):
        raise ValueError(
            "Compiled collision masks differ from ground-only owner profile"
        )
    support = []
    for name in quality.support_links:
        body = model.body(name).id
        geoms = [g for g in collision if model.geom_bodyid[g] == body]
        if not geoms:
            raise ValueError("Every support link needs owned collision geometry")
        support.extend(geoms)
    data = mujoco.MjData(model)
    shifts, ground_minima = [], []
    for t in range(len(q)):
        address = meta["free_qpos_adr"]
        data.qpos[address : address + 3] = root_p[t]
        data.qpos[address + 3 : address + 7] = root_q[t]
        data.qpos[meta["qpos_adr"]] = q[t]
        mujoco.mj_kinematics(model, data)
        shift = quality.ground_clearance_m - min(
            geometry_lowest_z(model, data, g) for g in support
        )
        root_p[t, 2] += shift
        shifts.append(shift)
        ground_minima.append(
            min(geometry_lowest_z(model, data, g) for g in collision) + shift
        )
    speed = np.abs(np.diff(q, axis=0)) * fps / quality.time_scale
    checks = {
        "native_bounds": bool(
            np.all(q >= np.asarray(meta["q_lo"]) - 1e-10)
            and np.all(q <= np.asarray(meta["q_hi"]) + 1e-10)
        ),
        "landmark_position": bool(not pe.size or np.max(pe) <= quality.position_tolerance_m),
        "landmark_orientation": bool(not re.size or np.max(re) <= quality.orientation_tolerance_rad),
        "root_projection": bool(np.max(np.abs(shifts)) <= quality.max_root_shift_m),
        "joint_speed": bool(np.max(speed) <= quality.max_joint_speed_rad_s),
        "floor_clearance": bool(
            np.min(ground_minima) >= -quality.max_ground_penetration_m
        ),
    }
    output = {k: np.array(v, copy=True) for k, v in arrays.items()}
    output["root_p"] = root_p
    output["fps"] = np.asarray(fps / quality.time_scale)
    report = {
        "schema": "xmimic.nominal-reference-quality/v1",
        "contract": asdict(quality),
        "accepted": all(checks.values()),
        "checks": checks,
        "source_frames": len(q),
        "source_fps": fps,
        "reference_fps": float(output["fps"]),
        "root_z_shift_m": shifts,
        "max_abs_root_shift_m": float(np.max(np.abs(shifts))),
        "minimum_collision_geometry_z_m": float(np.min(ground_minima)),
        "max_joint_speed_rad_s": float(np.max(speed)),
        "max_position_residual_m": float(np.max(pe)) if pe.size else None,
        "max_orientation_residual_rad": float(np.max(re)) if re.size else None,
        "owner_scene_sha256": receipt["scene_sha256"],
        "owner_profile_sha256": receipt["profile_sha256"],
        "self_collision_qualified": False,
        "balance_qualified": False,
        "physical_qualified": False,
        "scope": "nominal ground-only simulation reference; whole-pose floor projection and whole-clock time dilation; no foot-contact labels or no-slip guarantee",
    }
    return output, report
