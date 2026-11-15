"""Make the flight phases of a reference motion dynamically consistent, for any floating-base body.

A kinematic retarget copies the human's flight, but a robot with other masses and limb lengths cannot
follow it in physics: in the air its centre of mass must fly a parabola under gravity and its angular
momentum about the CoM must stay constant. This step keeps the joint trajectory and rewrites only the
floating root inside each flight window:

1. flight windows: the lowest body point is above ``clearance_m`` for at least ``min_flight_s``;
2. the takeoff vertical speed the ballistic flight needs is compared with what the actuators can give
   (|d(CoM - support)/dq|_z . rated joint speed, summed); if it is more, the apex is lowered and the
   flight is time-warped so its duration follows from the achievable vertical speed;
3. root orientation is integrated so the angular momentum about the CoM stays constant; the constant
   keeps the reference's mean axis and its magnitude is searched so the landing orientation matches;
4. root translation puts the CoM exactly on the ballistic parabola between takeoff and landing CoM;
5. the splice is blended over ``blend_s`` on both sides.

Everything is read from the MuJoCo model (masses, inertias, kinematics) and a joint -> speed table.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.transform import Rotation, Slerp

G = 9.81


@dataclass
class FlightReport:
    start: int
    end: int  # last airborne frame (inclusive), in the input timeline
    new_end: int  # in the output timeline
    duration_s: float
    new_duration_s: float
    takeoff_vz_needed: float
    takeoff_vz_achievable: float
    time_warp: float
    com_residual_before_m: float
    com_residual_after_m: float
    angmom_drift_before: float  # max |L - mean L| / |mean L|
    angmom_drift_after: float
    landing_orientation_error_deg: float

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def _velocities(m, q: np.ndarray, dt: float) -> np.ndarray:
    v = np.zeros((len(q), m.nv))
    for t in range(len(q)):
        a, b = max(t - 1, 0), min(t + 1, len(q) - 1)
        if b > a:
            mujoco.mj_differentiatePos(m, v[t], dt * (b - a), q[a], q[b])
    return v


def com_and_angmom(m, q: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Whole-body CoM (T,3) and angular momentum about the CoM (T,3), world frame."""
    d = mujoco.MjData(m)
    v = _velocities(m, q, dt)
    root = m.jnt_bodyid[0]
    com, L = np.zeros((len(q), 3)), np.zeros((len(q), 3))
    for t in range(len(q)):
        d.qpos[:], d.qvel[:] = q[t], v[t]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        mujoco.mj_subtreeVel(m, d)
        com[t], L[t] = d.subtree_com[root], d.subtree_angmom[root]
    return com, L


def ballistic_residual(com: np.ndarray, dt: float) -> float:
    """Max distance of a CoM path from its best-fit parabola with acceleration (0, 0, -g)."""
    t = np.arange(len(com)) * dt
    y = com + np.outer(0.5 * G * t**2, [0, 0, 1])  # remove gravity: the rest must be linear in t
    A = np.stack([np.ones_like(t), t], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.max(np.linalg.norm(y - A @ coef, axis=1)))


def angmom_drift(L: np.ndarray) -> float:
    mean = L.mean(0)
    return float(np.max(np.linalg.norm(L - mean, axis=1)) / max(np.linalg.norm(mean), 1e-6))


def flight_windows(lowest: np.ndarray, fps: float, clearance_m: float = 0.03, min_flight_s: float = 0.08):
    air = lowest > clearance_m
    out, t = [], 0
    while t < len(air):
        if air[t]:
            s = t
            while t < len(air) and air[t]:
                t += 1
            if (t - s) / fps >= min_flight_s and s > 0 and t < len(air):
                out.append((s, t - 1))
        t += 1
    return out


def achievable_vz(m, q: np.ndarray, support_points: list[tuple[int, np.ndarray]], speeds: np.ndarray) -> float:
    """Upper bound of the CoM vertical speed relative to the support, from joint speed limits.

    ``support_points``: (body, world point) of the contacts at takeoff; ``speeds``: rad/s per dof (0 = none).
    """
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    jcom = np.zeros((3, m.nv))
    mujoco.mj_jacSubtreeCom(m, d, jcom, m.jnt_bodyid[0])
    best = 0.0
    for body, point in support_points:
        jp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, None, point, body)
        rel = (jcom - jp)[2, 6:]
        best = max(best, float(np.sum(np.abs(rel) * speeds[6:])))
    return best


def _root_omega(m, d, q, qvel_joints, L_target) -> np.ndarray:
    """Root angular velocity (root-local, qvel[3:6]) giving angular momentum ``L_target`` about the CoM."""
    root = m.jnt_bodyid[0]

    def L_of(v):
        d.qpos[:], d.qvel[:] = q, v
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        mujoco.mj_subtreeVel(m, d)
        return d.subtree_angmom[root].copy()

    base = np.zeros(m.nv)
    base[6:] = qvel_joints
    L0 = L_of(base)
    A = np.zeros((3, 3))
    for i in range(3):
        v = base.copy()
        v[3 + i] += 1.0
        A[:, i] = L_of(v) - L0
    return np.linalg.solve(A, L_target - L0)


def _integrate_orientation(m, q_seg: np.ndarray, dt: float, L_target: np.ndarray) -> np.ndarray:
    """Integrate the root orientation over a segment so that L about the CoM equals L_target."""
    d = mujoco.MjData(m)
    out = q_seg.copy()
    vj = _velocities(m, q_seg, dt)[:, 6:]
    for t in range(1, len(out)):
        w = _root_omega(m, d, out[t - 1], vj[t - 1], L_target)
        x = np.zeros(m.nv)
        x[3:6] = w * dt
        nxt = out[t - 1].copy()
        mujoco.mj_integratePos(m, nxt, x, 1.0)
        out[t, 3:7] = nxt[3:7]
    return out


def _warp(q: np.ndarray, factor: float) -> np.ndarray:
    """Resample a segment to round(len * factor) frames (joints linear, root quaternion slerp)."""
    n = len(q)
    n2 = max(2, int(round((n - 1) * factor)) + 1)
    src = np.linspace(0, n - 1, n2)
    out = np.empty((n2, q.shape[1]))
    for j in range(q.shape[1]):
        out[:, j] = np.interp(src, np.arange(n), q[:, j])
    rot = Slerp(np.arange(n), Rotation.from_quat(q[:, 3:7], scalar_first=True))
    out[:, 3:7] = rot(src).as_quat(scalar_first=True)
    return out


def make_flights_consistent(m, q: np.ndarray, fps: float, lowest_fn, speeds: np.ndarray,
                            support_sites: list[tuple[int, np.ndarray]], blend_s: float = 0.05,
                            clearance_m: float = 0.03) -> tuple[np.ndarray, list[FlightReport], np.ndarray]:
    """Return (new qpos, per-flight reports, time map new_frame -> source frame)."""
    dt = 1.0 / fps
    d = mujoco.MjData(m)
    lowest = np.empty(len(q))
    for t in range(len(q)):
        d.qpos[:] = q[t]
        mujoco.mj_kinematics(m, d)
        lowest[t] = lowest_fn(d)
    windows = flight_windows(lowest, fps, clearance_m)
    pieces, tmap, reports, cursor = [], [], [], 0
    for s, e in windows:
        a, b = s - 1, e + 1  # last contact frame before, first contact frame after
        pieces.append(q[cursor:a])
        tmap.append(np.arange(cursor, a, dtype=float))
        seg = q[a:b + 1].copy()
        com, L = com_and_angmom(m, seg, dt)
        res_before, drift_before = ballistic_residual(com[1:-1], dt), angmom_drift(L[1:-1])
        T = (b - a) * dt
        vz_need = (com[-1, 2] - com[0, 2]) / T + 0.5 * G * T
        # support at takeoff: corresponding sites near the floor in the last contact frame
        d.qpos[:] = q[a]
        mujoco.mj_kinematics(m, d)
        supports = []
        for body, local in support_sites:
            p = d.xpos[body] + d.xmat[body].reshape(3, 3) @ local
            if p[2] < lowest[a] + 0.08:
                supports.append((body, p))
        vz_max = achievable_vz(m, q[a], supports or [(support_sites[0][0], d.xpos[support_sites[0][0]].copy())],
                               speeds)
        warp = 1.0
        if vz_need > vz_max > 0:
            dz = com[-1, 2] - com[0, 2]
            disc = vz_max**2 - 2 * G * dz
            T_new = (vz_max + np.sqrt(max(disc, 0.0))) / G
            warp = T_new / T
        seg_w = _warp(seg, warp) if warp != 1.0 else seg
        T_new = (len(seg_w) - 1) * dt
        # orientation: constant L about the CoM, constant chosen so the landing orientation matches
        R_land = Rotation.from_quat(seg_w[-1, 3:7], scalar_first=True)
        _, Lw = com_and_angmom(m, seg_w, dt)
        Lc = Lw[1:-1].mean(0)

        def landing_err(Lt):
            o = _integrate_orientation(m, seg_w, dt, Lt)
            return (Rotation.from_quat(o[-1, 3:7], scalar_first=True) * R_land.inv()).as_rotvec(), o

        # the reference's mean angular momentum fixes the spin axis; its magnitude is chosen (bounded
        # scalar search) so the integrated landing orientation matches the reference landing
        res = minimize_scalar(lambda c: float(np.linalg.norm(landing_err(c * Lc)[0])), bounds=(0.5, 1.5),
                              method="bounded", options={"xatol": 1e-3})
        Lc = res.x * Lc
        # then all three components (a twisting flip's axis is not the mean axis exactly)
        scale = float(np.linalg.norm(Lc))
        nm = minimize(lambda z: float(np.linalg.norm(landing_err(Lc + scale * z)[0])), np.zeros(3),
                      method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-3, "maxfev": 120,
                                                     "initial_simplex": 0.08 * np.vstack([np.zeros(3), np.eye(3)])})
        Lc = Lc + scale * nm.x
        r_final, seg_o = landing_err(Lc)
        # translation: CoM on the ballistic parabola between the (warped) takeoff and landing CoM
        com_o, _ = com_and_angmom(m, seg_o, dt)
        tt = np.arange(len(seg_o)) * dt
        v0 = (com_o[-1] - com_o[0]) / T_new + np.array([0, 0, 0.5 * G * T_new])
        target = com_o[0] + np.outer(tt, v0) - np.outer(0.5 * G * tt**2, [0, 0, 1])
        seg_o[:, :3] += target - com_o
        com_a, L_a = com_and_angmom(m, seg_o, dt)
        pieces.append(seg_o)
        tmap.append(np.linspace(a, b, len(seg_o)))
        cursor = b + 1
        reports.append(FlightReport(
            start=s, end=e, new_end=0, duration_s=T, new_duration_s=T_new, takeoff_vz_needed=float(vz_need),
            takeoff_vz_achievable=float(vz_max), time_warp=float(warp), com_residual_before_m=res_before,
            com_residual_after_m=ballistic_residual(com_a[1:-1], dt), angmom_drift_before=drift_before,
            angmom_drift_after=angmom_drift(L_a[1:-1]),
            landing_orientation_error_deg=float(np.degrees(np.linalg.norm(r_final)))))
    pieces.append(q[cursor:])
    tmap.append(np.arange(cursor, len(q), dtype=float))
    out = np.concatenate(pieces)
    tm = np.concatenate(tmap)
    # blend each splice boundary: gaussian-smooth the root translation in a window around it
    nb = max(1, int(round(blend_s * fps)))
    for rep, (s_, e_) in zip(reports, windows):
        k0 = int(np.searchsorted(tm, s_ - 1))
        k1 = int(np.searchsorted(tm, e_ + 1))
        rep.new_end = k1 - 1
        for k in (k0, k1):
            lo, hi = max(k - nb, 0), min(k + nb, len(out) - 1)
            src = out[:, :3].copy()
            for i in range(lo, hi + 1):
                w = np.exp(-0.5 * ((np.arange(i - nb, i + nb + 1)) - i) ** 2 / (nb / 2) ** 2)
                idx = np.clip(np.arange(i - nb, i + nb + 1), 0, len(out) - 1)
                out[i, :3] = (w[:, None] * src[idx]).sum(0) / w.sum()
        # report the final (blended) flight interior
        com_f, L_f = com_and_angmom(m, out[k0:k1 + 1], dt)
        rep.com_residual_after_m = ballistic_residual(com_f[1 + nb:-1 - nb] if k1 - k0 > 2 * nb + 4 else com_f[1:-1], dt)
        rep.angmom_drift_after = angmom_drift(L_f[1 + nb:-1 - nb] if k1 - k0 > 2 * nb + 4 else L_f[1:-1])
    return out, reports, tm
