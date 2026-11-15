"""Apply the dynamically consistent flight step to retargeted clips (see xmimic/retarget_flight.py).

    python -m tools.flight_consistency --body-document BODY.json --scene SCENE.xml \
        --correspondence tools/soma_correspondence.json --ratings RATINGS.json \
        [--speed-override DM10010=10.47] --joint-names META --embodiment-id ID \
        --output OUT_RUN IN_RUN/<clip> ...

Reads IN_RUN/<clip>/solution.npz (full qpos), writes OUT_RUN/<clip>/{motion,solution,series}.npz and
metrics.json with a ``flight`` list (before/after CoM parabola residual, angular-momentum drift,
needed vs achievable takeoff speed, time warp). Clips without a flight are copied through unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from xmimic.retarget_flight import make_flights_consistent
from xmimic.retarget_judge import motion_metrics
from xmimic.retarget_whole_body import WholeBodyRetargeter


def dof_speeds(m, ratings: dict, override: dict[str, float]) -> np.ndarray:
    speeds = np.zeros(m.nv)
    for j in range(m.njnt):
        model = ratings["joints"].get(m.joint(j).name)
        if model is None:
            continue
        v = override.get(model.split()[0], ratings["models"][model]["rated_rad_s"])
        speeds[m.jnt_dofadr[j]] = v
    return speeds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--body-document", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--correspondence", required=True)
    p.add_argument("--ratings", required=True)
    p.add_argument("--speed-override", action="append", default=[], help="MODEL=rad_s (declared assumption)")
    p.add_argument("--joint-names", required=True)
    p.add_argument("--embodiment-id", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("runs", nargs="+")
    args = p.parse_args()
    m = mujoco.MjModel.from_xml_path(args.scene)
    doc = json.loads(Path(args.body_document).read_text())
    corr = json.loads(Path(args.correspondence).read_text())["sites"]
    r = WholeBodyRetargeter.from_document(m, doc, corr)
    ratings = json.loads(Path(args.ratings).read_text())
    override = {k: float(v) for k, v in (s.split("=") for s in args.speed_override)}
    speeds = dof_speeds(m, ratings, override)
    names = json.loads(Path(args.joint_names).read_text())["joint_names"]
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
    supports = [(s.body, s.offset) for s in r.sites.values()]
    for run in map(Path, args.runs):
        sol = dict(np.load(run / "solution.npz"))
        before = json.loads((run / "metrics.json").read_text())
        fps = float(before["fps"])
        q, reports, tm = make_flights_consistent(m, sol["qpos"], fps, r._lowest_point, speeds, supports)
        # carry the per-frame human targets through the same time map (for the judge's side-by-side)
        tp = sol["target_pos"]
        tp2 = np.stack([np.stack([np.interp(tm, np.arange(len(tp)), tp[:, i, k]) for k in range(3)], -1)
                        for i in range(tp.shape[1])], 1)
        out = Path(args.output) / run.name
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "motion.npz", root_p=q[:, :3], root_quat=q[:, 3:7], q=q[:, adr], fps=np.asarray(fps),
                 joint_names=np.asarray(names), embodiment_id=np.asarray(args.embodiment_id))
        sol.update(qpos=q, target_pos=tp2, frames=np.arange(len(q)), ground_shift=np.zeros(len(q)),
                   time_map=tm)
        for k in ("raw_qpos", "target_rot", "position_error_m", "orientation_error_deg"):
            sol.pop(k, None)
        np.savez(out / "solution.npz", **sol)
        metrics = motion_metrics(r, q, tp2, [str(x) for x in sol["site_names"]], fps)
        series = metrics.pop("_series")
        np.savez(out / "series.npz", **{k: np.asarray(v) for k, v in series.items()})
        low_in_flight = [float(series["low"][rep.start:rep.new_end + 1].min()) for rep in reports]
        metrics.update({
            "clip": before["clip"], "source_sha256": before["source_sha256"], "fps": fps,
            "window": before["window"], "cropped": before["cropped"], "source_frames": before["source_frames"],
            "raw_direction_deg": before.get("raw_direction_deg"), "solver": before.get("solver"),
            "flight": [dict(rep.as_dict(), lowest_point_in_flight_m=lo) for rep, lo in zip(reports, low_in_flight)],
            "flight_step": {"speed_override": override, "ratings": ratings.get("source"),
                            "parent_run": str(run)},
            "motion_sha256": hashlib.sha256((out / "motion.npz").read_bytes()).hexdigest(),
        })
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        print(json.dumps({"clip": before["clip"], "flight": metrics["flight"]}), flush=True)


if __name__ == "__main__":
    main()
