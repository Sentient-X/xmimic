"""Retain and render a bounded CPU rollout of untrained nominal position servos.

The owner bundle supplies all geometry, gains and forces. This is neither a learned
policy nor a retargeted reference. A fall is retained, never tuned away by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from xmimic.embodiment_bundle import verify_nominal_receipt


def _font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.truetype("DejaVuSans.ttf", size)


def diagnose(robot: Path, output: Path, *, seconds: float = 2.0):
    """Keep every physics step and display actual visual meshes at 30 video fps."""
    if not math.isfinite(seconds) or not 0 < seconds <= 10:
        raise ValueError("CPU diagnostic duration must be in (0, 10] seconds")
    output = output.resolve()
    if output.exists() or "raw" in output.parts:
        raise ValueError("Use a new derived output directory outside raw")
    receipt = verify_nominal_receipt(robot)
    model = mujoco.MjModel.from_xml_path(str(robot / "scene.xml"))
    profile = receipt["profile"]
    if not 0 < model.opt.timestep <= 0.02:
        raise ValueError("Unsupported diagnostic physics timestep")
    steps = math.ceil(seconds / model.opt.timestep)
    if steps > 10000:
        raise ValueError("Diagnostic exceeds 10,000 CPU steps")
    names = receipt["joint_names"]
    joint_ids = [model.joint(name).id for name in names]
    qadr, dadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    actuator_ids = []
    for joint in joint_ids:
        found = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
        if len(found) != 1:
            raise ValueError("Expected exactly one nominal position servo per joint")
        actuator_ids.append(int(found[0]))
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("Expected one nominal floating root")
    fq = int(model.jnt_qposadr[free[0]])
    data = mujoco.MjData(model)
    defaults = np.asarray([joint["default_position"] for joint in profile["joints"]])
    data.qpos[qadr] = defaults
    data.ctrl[actuator_ids] = defaults
    mujoco.mj_forward(model, data)
    output.mkdir(parents=True)
    qpos, qvel, controls, times, contacts, distances, forces, warnings = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for step in range(steps + 1):
        if step:
            mujoco.mj_step(model, data)
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        controls.append(data.ctrl.copy())
        times.append(float(data.time))
        contacts.append(int(data.ncon))
        distances.append(
            min([float(c.dist) for c in data.contact[: data.ncon]] + [0.0])
        )
        forces.append(data.actuator_force.copy())
        warnings.append([int(w.number) for w in data.warning])
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            break  # Retain the numerical failure instead of presenting a fresh restart.
    qpos, qvel, controls = np.asarray(qpos), np.asarray(qvel), np.asarray(controls)
    trajectory = output / "trajectory.npz"
    np.savez_compressed(
        trajectory,
        qpos=qpos,
        qvel=qvel,
        ctrl=controls,
        time=np.asarray(times),
        contacts=np.asarray(contacts),
        minimum_contact_distance=np.asarray(distances),
        actuator_force=np.asarray(forces),
        warnings=np.asarray(warnings),
        embodiment_id=np.asarray(receipt["embodiment_id"]),
        scene_sha256=np.asarray(receipt["scene_sha256"]),
        joint_names=np.asarray(names),
    )
    finite = bool(np.isfinite(qpos).all() and np.isfinite(qvel).all())
    report = {
        "embodiment_id": receipt["embodiment_id"],
        "scene_sha256": receipt["scene_sha256"],
        "nominal_profile_sha256": receipt["profile_sha256"],
        "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        "controller": "untrained constant nominal position targets from owner profile",
        "physics_steps": len(times) - 1,
        "timestep_s": float(model.opt.timestep),
        "simulated_seconds": times[-1],
        "finite_every_step": finite,
        "initial_root_position": qpos[0, fq : fq + 3].tolist(),
        "final_root_position": qpos[-1, fq : fq + 3].tolist(),
        "final_root_quaternion_wxyz": qpos[-1, fq + 3 : fq + 7].tolist(),
        "max_contacts": max(contacts),
        "minimum_contact_distance_m": min(distances),
        "max_native_velocity_rad_s": float(np.max(np.abs(qvel[:, dadr]))),
        "max_actuator_force_Nm": float(np.max(np.abs(forces))),
        "warning_counts": warnings[-1],
        "mujoco_version": mujoco.__version__,
        "scope": "CPU physics baseline; no learned policy, balance or physical qualification",
        "training_performed": False,
    }
    (output / "diagnostic.json").write_text(json.dumps(report, indent=2))
    if not finite:
        raise ValueError(
            "Nonfinite rollout retained; refusing to render it as a valid trajectory"
        )
    render(model, qpos, qvel, times, contacts, distances, fq, output)
    return report


def render(model, qpos, qvel, times, contacts, distances, fq, output):
    width, height, image_height = 960, 720, 580
    model.vis.global_.offwidth, model.vis.global_.offheight = width, image_height
    model.vis.headlight.ambient[:] = 0.35
    model.vis.headlight.diffuse[:] = 0.65
    model.vis.headlight.specular[:] = 0.08
    model.geom_rgba[model.geom_group == 1] = [0.62, 0.72, 0.80, 1]
    ground = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    model.geom_group[ground] = 1
    model.geom_rgba[ground] = [0.10, 0.13, 0.17, 1]
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[1] = 1
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.30, 0, 0.65]
    camera.azimuth, camera.elevation, camera.distance = 55, -12, 2.8
    data = mujoco.MjData(model)
    count = max(2, round(times[-1] * 30) + 1)
    indices = np.rint(np.linspace(0, len(times) - 1, count)).astype(int)
    selected = {0, len(indices) // 2, len(indices) - 1}
    sheet = Image.new("RGB", (1440, 360), (14, 18, 23))
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        "30",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output / "untrained-pd-baseline.mp4"),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    renderer = mujoco.Renderer(model, height=image_height, width=width, max_geom=10000)
    slot = 0
    try:
        for frame, index in enumerate(indices):
            data.qpos[:] = qpos[index]
            data.qvel[:] = qvel[index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            image = Image.new("RGB", (width, height), (14, 18, 23))
            image.paste(Image.fromarray(renderer.render()), (0, 66))
            draw = ImageDraw.Draw(image)
            draw.text(
                (20, 10),
                "Untrained nominal PD baseline",
                font=_font(27),
                fill=(215, 230, 240),
            )
            draw.text(
                (20, 43),
                "Actual CPU physics · fixed position targets · no learned policy",
                font=_font(16),
                fill=(170, 195, 215),
            )
            draw.text(
                (20, 657),
                f"t={times[index]:.3f}s   root height={qpos[index, fq + 2]:.3f}m   contacts={contacts[index]}",
                font=_font(20),
                fill=(220, 230, 240),
            )
            draw.text(
                (20, 690),
                f"Deepest contact={distances[index] * 1000:.1f}mm · motion is retained, not corrected",
                font=_font(15),
                fill=(245, 185, 140),
            )
            process.stdin.write(np.asarray(image).tobytes())
            if frame in selected:
                sheet.paste(image.resize((480, 360)), (480 * slot, 0))
                image.save(output / f"baseline-{slot}.png")
                slot += 1
    finally:
        renderer.close()
        process.stdin.close()
        status = process.wait()
    if status:
        raise RuntimeError("Video encoder failed; physics trajectory remains retained")
    sheet.save(output / "contact-sheet.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.robot, args.output, seconds=args.seconds), indent=2))


if __name__ == "__main__":
    main()
