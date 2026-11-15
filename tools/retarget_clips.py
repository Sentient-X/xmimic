"""Retarget SOMA BVH clips onto one body with the whole-body site-calibrated solver.

    python -m tools.retarget_clips --body-document BODY.json --scene SCENE.xml \
        --calibration-source CAL.bvh --calibration-frame 0 \
        --correspondence tools/soma_correspondence.json --output OUT_DIR CLIP.bvh [CLIP.bvh ...]

Per clip it writes OUT_DIR/<clip>/motion.npz (the trainer's retargeted_npz boundary),
solution.npz (targets, residuals, raw solve) and metrics.json. Nothing is cropped: an optional
--window START:END is recorded in metrics.json. Private source/motion bytes stay outside git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from xmimic.retarget_source import parse_soma_bvh
from xmimic.retarget_whole_body import WholeBodyRetargeter
from xmimic.retarget_judge import motion_metrics


def build(args) -> tuple[WholeBodyRetargeter, dict, np.ndarray]:
    corr = json.loads(Path(args.correspondence).read_text())
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    limits = {}
    if getattr(args, "retarget_limits", None):
        # Declared retarget-side limits: the part of a body range a reference motion may use (e.g. an
        # elbow's swing past straight that the robot never uses). They may only narrow the body's range.
        limits = json.loads(Path(args.retarget_limits).read_text())["ranges"]
        for name, (lo, hi) in limits.items():
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            body_lo, body_hi = model.jnt_range[j]
            if lo < body_lo - 1e-6 or hi > body_hi + 1e-6:
                raise SystemExit(f"retarget limit for {name} widens the body range")
            model.jnt_range[j] = (lo, hi)
    doc = json.loads(Path(args.body_document).read_text())
    r = WholeBodyRetargeter.from_document(model, doc, corr["sites"])
    world = np.asarray(corr["world_from_source"], float)
    cal = parse_soma_bvh(Path(args.calibration_source).read_bytes(), world_from_source=world)
    report = r.calibrate(cal, args.calibration_frame)
    report["source_sha256"] = cal.source_sha256
    report["retarget_limits"] = limits
    return r, report, world


def retarget_one(r: WholeBodyRetargeter, world: np.ndarray, path: Path, out: Path, embodiment_id: str,
                 joint_names: list[str], window: tuple[int, int] | None, cutoff_hz: float) -> dict:
    clip = parse_soma_bvh(path.read_bytes(), world_from_source=world)
    start, end = window if window else (0, clip.frame_count)
    frames = np.arange(start, min(end, clip.frame_count))
    t0 = time.time()
    sol = r.solve(clip, frames)
    q = r.ground_and_smooth(clip, sol, clip.fps, cutoff_hz=cutoff_hz)
    seconds = time.time() - t0
    m = r.model
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "motion.npz", root_p=q[:, :3], root_quat=q[:, 3:7], q=q[:, adr], fps=np.asarray(clip.fps),
             joint_names=np.asarray(joint_names), embodiment_id=np.asarray(embodiment_id))
    hip0 = r._source(clip, int(frames[0]), r.root_site)[0][:2].copy()
    targets = [r.targets(clip, int(f), hip0) for f in frames]
    tp = np.asarray([[t[n][0] for n in sol.site_names] for t in targets])
    tr = np.asarray([[t[n][1] for n in sol.site_names] for t in targets])
    tp[:, :, 2] += r.ground_shift[:, None]  # targets in the grounded frame
    np.savez(out / "solution.npz", qpos=q, raw_qpos=sol.qpos, frames=frames, target_pos=tp, target_rot=tr,
             site_names=np.asarray(sol.site_names), position_error_m=sol.position_error_m,
             orientation_error_deg=sol.orientation_error_deg, ground_shift=r.ground_shift)
    human = np.asarray([[clip.world_positions[f, clip.joint_index(r.correspondence[n])] for n in sol.site_names]
                        for f in frames])
    metrics = motion_metrics(r, q, tp, sol.site_names, clip.fps, human_pos=human)
    series = metrics.pop("_series")
    np.savez(out / "series.npz", **{k: np.asarray(v) for k, v in series.items()})
    metrics.update({
        "clip": path.stem, "source_sha256": clip.source_sha256, "source_frames": clip.frame_count,
        "window": [int(frames[0]), int(frames[-1]) + 1], "cropped": bool(len(frames) != clip.frame_count),
        "fps": clip.fps, "solve_seconds": round(seconds, 2), "mean_nfev": float(sol.nfev.mean()),
        "motion_sha256": hashlib.sha256((out / "motion.npz").read_bytes()).hexdigest(),
        "solve_site_position_error_max_m": dict(zip(sol.site_names, sol.position_error_m.max(0).round(4).tolist())),
        "solver": {"max_joint_speed_rad_s": r.max_joint_speed, "cutoff_hz": cutoff_hz,
                   "retarget_limits": getattr(r, "retarget_limits", {})},
        "ground_shift_range_m": [float(r.ground_shift.min()), float(r.ground_shift.max())],
    })
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--body-document", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--embodiment-id", required=True)
    p.add_argument("--joint-names", required=True, help="JSON file with the body's native joint names (meta.json)")
    p.add_argument("--calibration-source", required=True)
    p.add_argument("--calibration-frame", type=int, default=0)
    p.add_argument("--correspondence", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--window", default=None, help="START:END source frames (recorded, never silent)")
    p.add_argument("--cutoff-hz", type=float, default=12.0)
    p.add_argument("--retarget-limits", default=None, help="JSON {ranges: {joint: [lo, hi]}}, narrowing only")
    p.add_argument("clips", nargs="+")
    args = p.parse_args()
    r, report, world = build(args)
    joint_names = json.loads(Path(args.joint_names).read_text())["joint_names"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(report, indent=1))
    window = tuple(int(x) for x in args.window.split(":")) if args.window else None
    r.retarget_limits = report["retarget_limits"]
    for clip in args.clips:
        path = Path(clip)
        metrics = retarget_one(r, world, path, out / path.stem, args.embodiment_id, joint_names, window,
                               args.cutoff_hz)
        print(json.dumps({k: metrics[k] for k in ("clip", "solve_seconds", "screen")}), flush=True)


if __name__ == "__main__":
    main()
