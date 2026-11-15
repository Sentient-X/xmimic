"""Judging a whole-body retarget: numbers that screen first, pictures that a person (or agent) reads.

Everything is derived from the retargeter's site tree, so it works for any body:
segment colours come from the side of each site in the calibration pose (left warm, right cool,
centre neutral); robot links take the colour of the segment they carry.

Screen thresholds are declared here, before any result is seen, and are never widened afterwards.
"""

from __future__ import annotations


import mujoco
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

SCREEN = {
    "direction_p99_deg": 15.0,  # articulated segment direction vs scaled-human target, 99th percentile
    "direction_max_deg": 35.0,
    "floor_penetration_min_m": -0.01,
    "contact_slide_p95_mps": 0.30,  # horizontal speed of a contacting foot site
    "jitter_hf_max_deg": 0.5,  # rms of joint angle above 8 Hz
}
CONTACT_Z_M = 0.025


def _site_positions(r, q: np.ndarray) -> np.ndarray:
    m = r.model
    d = mujoco.MjData(m)
    out = np.empty((len(q), len(r.sites), 3))
    names = list(r.sites)
    for t, x in enumerate(q):
        d.qpos[:] = x
        mujoco.mj_kinematics(m, d)
        for i, n in enumerate(names):
            out[t, i] = r._site_pose(d, n)[0]
    return out


def _seg_dist(p1, q1, p2, q2) -> float:
    """Minimum distance between segments p1q1 and p2q2."""
    d1, d2, rr = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ rr
    c, b = d1 @ rr, d1 @ d2
    denom = a * e - b * b
    s = np.clip((b * f - c * e) / denom, 0, 1) if denom > 1e-12 else 0.0
    t = (b * s + f) / e if e > 1e-12 else 0.0
    if t < 0:
        t, s = 0.0, np.clip(-c / a, 0, 1) if a > 1e-12 else 0.0
    elif t > 1:
        t, s = 1.0, np.clip((b - c) / a, 0, 1) if a > 1e-12 else 0.0
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def motion_metrics(r, q: np.ndarray, target_pos: np.ndarray, site_names: list[str], fps: float,
                   human_pos: np.ndarray | None = None) -> dict:
    """``human_pos`` (T, sites, 3): unscaled source joint positions, for the raw (uncalibrated) direction metric
    used by earlier URR evidence: angle between robot site pair and the corresponding source joint pair."""
    m = r.model
    names = list(r.sites)
    assert names == list(site_names)
    sp = _site_positions(r, q)
    idx = {n: i for i, n in enumerate(names)}
    segs = [s for s in r.segments if not s.rigid]
    seg_err = np.zeros((len(q), len(segs)))
    for k, s in enumerate(segs):
        a = sp[:, idx[s.child]] - sp[:, idx[s.parent]]
        b = target_pos[:, idx[s.child]] - target_pos[:, idx[s.parent]]
        cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        seg_err[:, k] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    raw_err = None
    if human_pos is not None:
        raw_err = np.zeros_like(seg_err)
        for k, s in enumerate(segs):
            a = sp[:, idx[s.child]] - sp[:, idx[s.parent]]
            b = human_pos[:, idx[s.child]] - human_pos[:, idx[s.parent]]
            cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
            raw_err[:, k] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    # joints
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    jn = [m.joint(j).name for j in hinge]
    qa = np.asarray([q[:, m.jnt_qposadr[j]] for j in hinge]).T
    lo = np.asarray([m.jnt_range[j][0] for j in hinge])
    hi = np.asarray([m.jnt_range[j][1] for j in hinge])
    sat = (qa - lo < np.radians(1.0)) | (hi - qa < np.radians(1.0))
    dt = 1.0 / fps
    qd = np.gradient(qa, dt, axis=0)
    # high-frequency residual (shiver): joint angle content above 8 Hz
    if len(qa) > 30:
        b, a = butter(2, 8.0 / (fps / 2))
        hf = np.degrees(np.sqrt(np.mean((qa - filtfilt(b, a, qa, axis=0)) ** 2, axis=0)))
    else:
        hf = np.zeros(len(hinge))
    # inverse-dynamics torque proxy along the kinematic trajectory
    d = mujoco.MjData(m)
    vel = np.zeros((len(q), m.nv))
    for t in range(len(q)):
        a0, a1 = max(t - 1, 0), min(t + 1, len(q) - 1)
        mujoco.mj_differentiatePos(m, vel[t], dt * (a1 - a0), q[a0], q[a1])
    acc = np.gradient(vel, dt, axis=0)
    tau = np.zeros((len(q), len(hinge)))
    root_force = np.zeros(len(q))
    mass = float(np.sum(m.body_mass))
    for t in range(len(q)):
        d.qpos[:], d.qvel[:], d.qacc[:] = q[t], vel[t], acc[t]
        mujoco.mj_inverse(m, d)
        tau[t] = [d.qfrc_inverse[m.jnt_dofadr[j]] for j in hinge]
        root_force[t] = np.linalg.norm(d.qfrc_inverse[:3])
    # floor and contact slide
    low = np.empty(len(q))
    for t in range(len(q)):
        d.qpos[:] = q[t]
        mujoco.mj_kinematics(m, d)
        low[t] = r._lowest_point(d)
    leaves = [n for n in names if not any(s.parent == n for s in r.segments)]
    slide = []
    for n in leaves:
        p = sp[:, idx[n]]
        v = np.linalg.norm(np.gradient(p[:, :2], dt, axis=0), axis=1)
        contact = p[:, 2] - np.maximum(low, 0) < CONTACT_Z_M + 0.03
        contact &= p[:, 2] < 0.08
        slide.extend(v[contact].tolist())
    slide = np.asarray(slide) if slide else np.zeros(1)
    # self proximity between non-adjacent articulated segments
    close = np.zeros(len(q), bool)
    pairs = [(i, k) for i in range(len(segs)) for k in range(i + 1, len(segs))
             if not {segs[i].parent, segs[i].child} & {segs[k].parent, segs[k].child}]
    for t in range(0, len(q), 2):
        for i, k in pairs:
            si, sk = segs[i], segs[k]
            dd = _seg_dist(sp[t, idx[si.parent]], sp[t, idx[si.child]], sp[t, idx[sk.parent]], sp[t, idx[sk.child]])
            if dd < 0.04:
                close[t] = True
                break
    up = Rotation.from_quat(q[:, 3:7], scalar_first=True).apply([0, 0, 1])
    tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
    screen_values = {
        "direction_p99_deg": float(np.percentile(seg_err, 99)),
        "direction_max_deg": float(seg_err.max()),
        "floor_penetration_min_m": float(low.min()),
        "contact_slide_p95_mps": float(np.percentile(slide, 95)),
        "jitter_hf_max_deg": float(hf.max()),
    }
    passed = (screen_values["direction_p99_deg"] <= SCREEN["direction_p99_deg"]
              and screen_values["direction_max_deg"] <= SCREEN["direction_max_deg"]
              and screen_values["floor_penetration_min_m"] >= SCREEN["floor_penetration_min_m"]
              and screen_values["contact_slide_p95_mps"] <= SCREEN["contact_slide_p95_mps"]
              and screen_values["jitter_hf_max_deg"] <= SCREEN["jitter_hf_max_deg"])
    peak_speed = np.abs(qd).max(0)
    peak_tau = np.abs(tau).max(0)
    return {
        "screen": {"passed": bool(passed), "values": screen_values, "thresholds": SCREEN},
        "segments": [f"{s.parent}->{s.child}" for s in segs],
        "direction_p99_per_segment_deg": dict(zip([f"{s.parent}->{s.child}" for s in segs],
                                                  np.percentile(seg_err, 99, axis=0).round(2).tolist())),
        "joint_peak_speed_rad_s": dict(zip(jn, peak_speed.round(3).tolist())),
        "joint_peak_torque_proxy_nm": dict(zip(jn, peak_tau.round(1).tolist())),
        "joint_limit_saturation_fraction": dict(zip(jn, sat.mean(0).round(4).tolist())),
        "root_external_force_peak_body_weights": float(root_force.max() / (mass * 9.81)),
        "root_tilt_max_deg": float(tilt.max()),
        "raw_direction_deg": None if raw_err is None else {
            "max": float(raw_err.max()), "p99": float(np.percentile(raw_err, 99)),
            "median": float(np.median(raw_err))},
        "root_height_range_m": [float(q[:, 2].min()), float(q[:, 2].max())],
        "airborne_fraction": float(np.mean(low > 0.03)),
        "self_proximity_fraction": float(close[::2].mean()),
        "_series": {"seg_err": seg_err, "sat": sat, "qd": qd, "tau": tau, "low": low, "tilt": tilt,
                    "root_force": root_force / (mass * 9.81), "joint_names": jn, "hf": hf,
                    "close": close},
    }


def strip_series(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "_series"}
