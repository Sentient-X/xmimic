"""Render byte-bound SOMA and canonical robot candidate motion with MuJoCo meshes.

This is a kinematic diagnostic, not a simulated policy rollout. Source and robot
use the same source-frame clock; the video may downsample display frames only.
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

from xmimic.model import load_model
from xmimic.retarget_source import parse_soma_bvh
from xmimic.retarget_workflow import recipe_from_json


WIDTH, HEIGHT, PANEL_H, TOP = 1280, 800, 510, 66


def _font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.truetype("DejaVuSans.ttf", size)


def display_indices(frames, source_fps, video_fps):
    if (
        frames < 2
        or not math.isfinite(source_fps)
        or source_fps <= 0
        or not math.isfinite(video_fps)
        or video_fps <= 0
    ):
        raise ValueError("Positive frame rates and a complete source clip required")
    count = max(2, math.ceil(frames / source_fps * video_fps))
    indices = np.minimum(
        np.rint(np.arange(count) * source_fps / video_fps).astype(int), frames - 1
    )
    indices[-1] = frames - 1
    return indices


def limit_hits(q, lower, upper, active, *, margin_rad=math.radians(0.5)):
    q = np.asarray(q)
    return (
        (q - np.asarray(lower) <= margin_rad) | (np.asarray(upper) - q <= margin_rad)
    ) & np.asarray(active)


def add_geom(scene, kind, position, size, color, *, end=None):
    if scene.ngeom >= scene.maxgeom:
        raise ValueError("Diagnostic scene exceeded its geometry capacity")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        kind,
        np.asarray(size, dtype=float),
        np.asarray(position, dtype=float),
        np.eye(3).ravel(),
        np.asarray(color, np.float32),
    )
    if end is not None:
        mujoco.mjv_connector(
            geom,
            kind,
            size[0],
            np.asarray(position, dtype=float),
            np.asarray(end, dtype=float),
        )
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1


def render(source_path, recipe_path, robot_path, cache_path, output, *, video_fps=30):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(source_path).read_bytes()
    recipe = recipe_from_json(Path(recipe_path).read_bytes())
    clip = parse_soma_bvh(source, world_from_source=recipe.world_from_source)
    manifest = json.loads((Path(cache_path) / "manifest.json").read_text())
    if manifest["identity"]["source_sha256"] != hashlib.sha256(source).hexdigest():
        raise ValueError("Source bytes differ from the retained candidate")
    clips = list(manifest["clips"].values())
    if len(clips) != 1:
        raise ValueError("Choose one complete clip cache")
    artifacts = clips[0]["artifacts"]

    def artifact(name):
        item = artifacts[name]
        path = Path(cache_path) / "blobs" / item["sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Candidate artifact checksum differs")
        return path

    candidate = artifact("motion.npz" if "motion.npz" in artifacts else "candidate.npz")
    report = json.loads(artifact("retarget.json").read_text())
    if report["recipe"] != json.loads(Path(recipe_path).read_bytes()):
        raise ValueError("Recipe differs from the candidate provenance")
    with np.load(candidate, allow_pickle=False) as data:
        positions = data["root_p"].copy()
        quats = data["root_quat"].copy()
        q = data["q"].copy()
        names = data["joint_names"].tolist()
        body_id = str(data["embodiment_id"].item())
        reference_fps = float(data["fps"])
    model, meta = load_model(Path(robot_path), motion_format="retargeted_npz")
    if (
        names != meta["joint_names"]
        or body_id != meta["embodiment_id"]
        or len(q) != clip.frame_count
    ):
        raise ValueError("Candidate body, coordinate order or source timebase differs")
    residual = np.asarray(report["geometric_result"]["position_residual_m"]) * 1000
    orientation = np.asarray(report["geometric_result"]["orientation_residual_rad"])
    active = [name not in dict(recipe.target.holds) for name in names]
    hits = limit_hits(q, meta["q_lo"], meta["q_hi"], active)
    joint_bodies = [model.jnt_bodyid[model.joint(name).id] for name in names]
    model.vis.global_.offwidth = WIDTH // 2
    model.vis.global_.offheight = PANEL_H
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.7
    robot_data = mujoco.MjData(model)
    source_model = mujoco.MjModel.from_xml_string(
        '<mujoco><visual><global offwidth="640" offheight="510"/><headlight ambient=".6 .6 .6"/></visual><worldbody><geom type="plane" size="8 8 .1" rgba=".18 .22 .28 1"/></worldbody></mujoco>'
    )
    source_data = mujoco.MjData(source_model)
    mujoco.mj_forward(source_model, source_data)
    robot_renderer = mujoco.Renderer(
        model, height=PANEL_H, width=WIDTH // 2, max_geom=10000
    )
    source_renderer = mujoco.Renderer(
        source_model, height=PANEL_H, width=WIDTH // 2, max_geom=1000
    )
    robot_option = mujoco.MjvOption()
    robot_option.geomgroup[:] = 0
    robot_option.geomgroup[1] = 1
    # Render declared visual meshes, falling back to collisions only for ground.
    ground = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    model.geom_group[ground] = 1
    camera = mujoco.MjvCamera()
    camera.azimuth = 135
    camera.elevation = -14
    source_camera = mujoco.MjvCamera()
    source_camera.azimuth = 135
    source_camera.elevation = -14
    root_index = clip.joint_index(recipe.source_root)
    target_names = [m.source for m in recipe.target.landmarks]
    sample_indices = display_indices(clip.frame_count, reference_fps, video_fps)
    worst = int(residual.max(axis=1).argmax())
    sheet_slots = set(
        np.linspace(0, len(sample_indices) - 1, 6).astype(int).tolist()
        + [int(np.abs(sample_indices - worst).argmin())]
    )
    sheet_frames = []
    ffmpeg = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(video_fps),
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
        str(output / "source-and-candidate.mp4"),
    ]
    process = subprocess.Popen(ffmpeg, stdin=subprocess.PIPE)
    max_res = max(5.0, float(residual.max()) * 1.05)
    try:
        for number, frame in enumerate(sample_indices):
            robot_data.qpos[meta["free_qpos_adr"] : meta["free_qpos_adr"] + 3] = (
                positions[frame]
            )
            robot_data.qpos[meta["free_qpos_adr"] + 3 : meta["free_qpos_adr"] + 7] = (
                quats[frame]
            )
            robot_data.qpos[meta["qpos_adr"]] = q[frame]
            mujoco.mj_kinematics(model, robot_data)
            mujoco.mj_camlight(model, robot_data)
            camera.lookat[:] = [*positions[frame, :2], 0.65]
            camera.distance = 2.25
            robot_renderer.update_scene(
                robot_data, camera=camera, scene_option=robot_option
            )
            for joint in np.flatnonzero(hits[frame]):
                add_geom(
                    robot_renderer.scene,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    robot_data.xpos[joint_bodies[joint]],
                    [0.018] * 3,
                    [1, 0.22, 0.12, 1],
                )
            human = clip.world_positions[frame]
            source_camera.lookat[:] = [*human[root_index, :2], 1.0]
            source_camera.distance = 3.15
            source_renderer.update_scene(source_data, camera=source_camera)
            for joint, parent in enumerate(clip.parents):
                if parent < 0 or clip.joint_names[parent] == "Root":
                    continue  # virtual global root is composed by FK, not drawn as an anatomical bone
                color = (
                    [0.15, 0.78, 0.88, 1]
                    if not clip.joint_names[joint].startswith("Right")
                    else [0.48, 0.64, 1, 1]
                )
                add_geom(
                    source_renderer.scene,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    human[parent],
                    [0.007] * 3,
                    color,
                    end=human[joint],
                )
            for name in target_names:
                add_geom(
                    source_renderer.scene,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    human[clip.joint_index(name)],
                    [0.022] * 3,
                    [1, 0.7, 0.12, 1],
                )
            image = Image.new("RGB", (WIDTH, HEIGHT), (17, 24, 34))
            draw = ImageDraw.Draw(image)
            image.paste(Image.fromarray(source_renderer.render()), (0, TOP))
            image.paste(Image.fromarray(robot_renderer.render()), (WIDTH // 2, TOP))
            draw.text(
                (20, 12), "SOMA human source", font=_font(24), fill=(230, 240, 255)
            )
            draw.text(
                (660, 12),
                f"Canonical target · {len(names)} joints",
                font=_font(24),
                fill=(230, 240, 255),
            )
            draw.text(
                (20, 42),
                "BVH native proportions · declared source-to-world basis",
                font=_font(13),
                fill=(170, 192, 212),
            )
            draw.text(
                (660, 42),
                "Actual MuJoCo meshes · candidate replay, not policy physics",
                font=_font(13),
                fill=(170, 192, 212),
            )
            y = TOP + PANEL_H + 10
            draw.text(
                (20, y),
                f"Source {frame / clip.fps:.3f}s · reference {frame / reference_fps:.3f}s · frame {frame + 1}/{clip.frame_count}",
                font=_font(19),
                fill=(225, 235, 248),
            )
            passed = bool(
                np.all(residual[frame] <= recipe.solver.position_tolerance_m * 1000)
                and np.all(
                    orientation[frame] <= recipe.solver.orientation_tolerance_rad
                )
            )
            draw.text(
                (660, y),
                f"Optimizer {recipe.solver.position_tolerance_m * 1000:g} mm gate: {'PASS' if passed else 'FAIL'}   |   near limits: {int(hits[frame].sum())}",
                font=_font(19),
                fill=(80, 225, 170) if passed else (255, 133, 108),
            )
            draw.text(
                (20, y + 28),
                "Position error mm  "
                + "  ".join(
                    f"{name}: {value:.1f}"
                    for name, value in zip(target_names, residual[frame])
                ),
                font=_font(15),
                fill=(215, 225, 240),
            )
            names_hit = [names[i] for i in np.flatnonzero(hits[frame])]
            label = ", ".join(names_hit) if names_hit else "none"
            draw.text(
                (20, y + 52),
                "Red markers: active joints within 0.5° of a native limit.  " + label,
                font=_font(13),
                fill=(255, 180, 155),
            )
            gx, gy, gw, gh = 30, y + 90, 1220, 70
            curve = residual.max(axis=1)
            poly = [
                (gx + i / (len(curve) - 1) * gw, gy + gh - v / max_res * gh)
                for i, v in enumerate(curve)
            ]
            draw.line(poly, fill=(255, 190, 70), width=2)
            gate_y = gy + gh - recipe.solver.position_tolerance_m * 1000 / max_res * gh
            draw.line((gx, gate_y, gx + gw, gate_y), fill=(240, 95, 100), width=1)
            marker = gx + frame / (len(curve) - 1) * gw
            draw.line((marker, gy, marker, gy + gh), fill=(220, 235, 255), width=2)
            draw.text(
                (gx, gy - 18),
                f"Full-clip maximum landmark error  (0–{max_res:.0f} mm); red line = optimizer goal",
                font=_font(12),
                fill=(175, 195, 215),
            )
            process.stdin.write(np.asarray(image).tobytes())
            if number in sheet_slots:
                sheet_frames.append((number, image.copy()))
            if number == 0:
                image.save(output / "first-frame.png")
    finally:
        process.stdin.close()
        status = process.wait()
        source_renderer.close()
        robot_renderer.close()
    if status:
        raise RuntimeError("Video encoder failed")
    sheet = Image.new("RGB", (WIDTH * 3 // 2, HEIGHT), (17, 24, 34))
    for index, (_, frame) in enumerate(sheet_frames[:6]):
        sheet.paste(
            frame.resize((WIDTH // 2, HEIGHT // 2)),
            ((index % 3) * (WIDTH // 2), (index // 3) * (HEIGHT // 2)),
        )
    sheet.save(output / "contact-sheet.png")
    receipt = {
        "schema": "xmimic.retarget-visualization/v1",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "embodiment_id": body_id,
        "source_frames": clip.frame_count,
        "source_fps": clip.fps,
        "reference_fps": reference_fps,
        "display_frames": len(sample_indices),
        "display_fps": video_fps,
        "source_frame_indices": sample_indices.tolist(),
        "all_source_frames_retained_in_candidate": True,
        "physical_simulation": False,
        "active_joint_limit_margin_rad": math.radians(0.5),
        "scope": "synchronized kinematic source/candidate diagnostic; raw source native proportions; original residual gate retained",
    }
    (output / "visualization.json").write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source", "recipe", "robot", "cache", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    print(
        json.dumps(
            render(
                args.source,
                args.recipe,
                args.robot,
                args.cache,
                args.output,
                video_fps=args.fps,
            )
        )
    )
