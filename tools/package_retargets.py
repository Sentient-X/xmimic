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
    args = p.parse_args()
    spec = json.loads(Path(args.judgments).read_text())
    out = Path(args.output)
    (out / "motions").mkdir(parents=True, exist_ok=True)
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
        if c["accepted"]:
            dst = out / "motions" / f"{metrics['clip']}.npz"
            shutil.copyfile(run / "motion.npz", dst)
            entry["motion"] = f"motions/{dst.name}"
            entry["motion_sha256"] = hashlib.sha256(dst.read_bytes()).hexdigest()
        entries.append(entry)
    manifest = {"schema": "xmimic.retarget-batch/v1", "body": spec["body"], "method": spec.get("method", {}),
                "accepted": sum(e["accepted"] for e in entries), "clips": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{manifest['accepted']} accepted of {len(entries)} -> {out}")
    if args.volume:
        subprocess.run(["modal", "volume", "create", args.volume], check=False, capture_output=True)
        subprocess.run(["modal", "volume", "put", "--force", args.volume, str(out), out.name], check=True)
        print(f"uploaded to modal volume {args.volume}:/{out.name}")


if __name__ == "__main__":
    main()
