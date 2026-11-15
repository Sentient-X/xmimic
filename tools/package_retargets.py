"""Package judged retargets for the trainer: accepted motion.npz files plus one manifest.

    python -m tools.package_retargets --judgments JUDGMENTS.json --output OUT_DIR [--volume xmimic-motions]

JUDGMENTS.json: {"body": {...identity...}, "clips": [{"run_dir": ".../<clip>", "accepted": true,
"judgment": "one line", "frames": "keyframes looked at", "mirror": false}, ...]}.
Only accepted clips are copied to OUT_DIR/motions/ (the trainer's retargeted_npz directory);
every judged clip, accepted or rejected, is listed in OUT_DIR/manifest.json with its metrics.
With --volume the directory is uploaded to that Modal Volume under its own name (private bytes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--judgments", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--volume", default=None)
    p.add_argument("--volume-path", default=None, help="destination inside the volume (default: OUT_DIR name)")
    p.add_argument("--ratings", default=None, help="JSON {models: {m: {peak_nm, rated_rad_s, no_load_rad_s}}, joints: {j: m}}")
    p.add_argument("--flat", action="store_true", help="motions directly in OUT_DIR (trainer's flat layout)")
    args = p.parse_args()
    spec = json.loads(Path(args.judgments).read_text())
    out = Path(args.output)
    motions = out if args.flat else out / "motions"
    motions.mkdir(parents=True, exist_ok=True)
    ratings = json.loads(Path(args.ratings).read_text()) if args.ratings else None
    entries = []
    for c in spec["clips"]:
        run = Path(c["run_dir"])
        metrics = json.loads((run / "metrics.json").read_text())
        entry = {
            "clip": metrics["clip"], "mirror": metrics["clip"].endswith("_M"), "accepted": bool(c["accepted"]),
            "judgment": c["judgment"], "frames_reviewed": c.get("frames", ""),
            "window": metrics["window"], "cropped": metrics["cropped"], "fps": metrics["fps"],
            "source_sha256": metrics["source_sha256"], "screen": metrics["screen"],
            "raw_direction_deg": metrics.get("raw_direction_deg"),
            "joint_peak_speed_rad_s": metrics["joint_peak_speed_rad_s"],
            "joint_peak_torque_proxy_nm": metrics["joint_peak_torque_proxy_nm"],
            "root_external_force_peak_body_weights": metrics["root_external_force_peak_body_weights"],
            "airborne_fraction": metrics["airborne_fraction"], "root_tilt_max_deg": metrics["root_tilt_max_deg"],
            "judge_images": str(run / "judge"),
        }
        if ratings:
            # peak joint speed over the rated (24 V) and no-load speed of the joint's actuator; >1 = aggressive
            speed = {}
            for joint, v in metrics["joint_peak_speed_rad_s"].items():
                model = ratings["models"].get(ratings["joints"].get(joint, ""), None)
                if model:
                    speed[joint] = {"rated": round(v / model["rated_rad_s"], 2),
                                    "no_load": round(v / model["no_load_rad_s"], 2)}
            entry["speed_over_rating"] = speed
            entry["joints_over_no_load"] = sorted(j for j, r in speed.items() if r["no_load"] > 1)
        if c["accepted"]:
            dst = motions / f"{metrics['clip']}.npz"
            shutil.copyfile(run / "motion.npz", dst)
            entry["motion"] = str(dst.relative_to(out))
            entry["motion_sha256"] = hashlib.sha256(dst.read_bytes()).hexdigest()
        entries.append(entry)
    manifest = {"schema": "xmimic.retarget-batch/v1", "body": spec["body"], "method": spec.get("method", {}),
                "ratings": ratings and ratings.get("source"),
                "accepted": sum(e["accepted"] for e in entries), "clips": entries}
    # the trainer reads every top-level file of a flat directory as a clip: keep the manifest beside it
    manifest_path = (out.parent / f"{out.name}.manifest.json") if args.flat else out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"{manifest['accepted']} accepted of {len(entries)} -> {out}")
    if args.volume:
        subprocess.run(["modal", "volume", "create", args.volume], check=False, capture_output=True)
        dest = args.volume_path or out.name
        subprocess.run(["modal", "volume", "put", "--force", args.volume, str(out), dest], check=True)
        print(f"uploaded to modal volume {args.volume}:/{dest}")


if __name__ == "__main__":
    main()
