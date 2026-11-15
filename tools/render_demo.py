"""Evaluate a checkpoint on every clip on the CPU, render real-time MP4s and write a metrics table and gallery.

    python tools/render_demo.py <ckpt.npz> --robot_dir <bundle> --motion_dir <clips> --out <dir> \\
        [--clips a,b] [--label "datasheet actuators"] [--run sentient-v1]

Per clip: one episode from the clip's first frame under the evaluation protocol (tools/rollout_cpu.py: MuJoCo's C
simulator, no randomization, no noise), then
  - metrics: fall (training termination triggered), mean tracked-body position error in world and relative to the root
    (while alive), the share of the clip survived;
  - <clip>.mp4: 2 s title card, then a 2 x 2 grid at real time (50 fps): 3/4 tracking view and side view, the simulated
    robot (left) beside the kinematic reference it tracks (right), with a ground grid.
metrics.json, metrics.csv and index.html (a local gallery) go to --out. Simulation only: nothing commands a robot.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rollout_cpu import rollout  # noqa: E402

W, H = 640, 360
FPS = 50
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default(size)


CELL_M = 0.003  # render-mesh vertex clustering cell: ~1 px of a 360 px tall view of the robot


def cluster_mesh(vertices, faces, cell):
    """Vertex-clustering decimation: weld every vertex to its grid cell's mean and drop collapsed or repeated faces.
    Robust on CAD triangle soup, where quadric decimation stalls."""
    keys, inverse = np.unique(np.floor(vertices / cell).astype(np.int64), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    welded = np.zeros((len(keys), 3))
    np.add.at(welded, inverse, vertices)
    welded /= np.bincount(inverse, minlength=len(keys))[:, None]
    f = inverse[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    f = np.unique(f, axis=0)
    return welded, f


def render_meshes(scene: Path, xml: str) -> str:
    """Point every mesh at a clustered copy (cached beside the bundle) for the render model only; the 2M-face CAD meshes
    take ~1 s per view on a Mac."""
    import trimesh

    cache = scene.parent.parent / f"{scene.parent.name}-render-meshes"
    cache.mkdir(exist_ok=True)

    def swap(match):
        element = match.group(0)
        name = re.search(r'file="([^"]+)"', element).group(1)
        scale = re.search(r'scale="([^"]+)"', element)
        unit = float(scale.group(1).split()[0]) if scale else 1.0
        src, dst = scene.parent / name, cache / Path(name).name
        if not dst.exists():
            mesh = trimesh.load(src, force="mesh", process=False)
            v, f = cluster_mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), CELL_M / unit)
            trimesh.Trimesh(v, f, process=False).export(dst)
        return element.replace(f'file="{name}"', f'file="{dst}"')

    return re.sub(r"<mesh [^>]*>", swap, xml)


def render_model(scene: Path, tint=None):
    """The training scene with a ground grid, sky and lights for rendering only (physics never uses this model)."""
    xml = render_meshes(scene, scene.read_text())
    assets = ('<texture name="sx_grid" type="2d" builtin="checker" rgb1="0.16 0.18 0.22" rgb2="0.11 0.12 0.15" '
              'width="512" height="512" mark="edge" markrgb="0.45 0.5 0.6"/>'
              '<material name="sx_grid" texture="sx_grid" texrepeat="1 1" texuniform="true" reflectance="0.15"/>'
              '<texture name="sx_sky" type="skybox" builtin="gradient" rgb1="0.32 0.38 0.48" rgb2="0.04 0.05 0.08" '
              'width="512" height="512"/>')
    xml = xml.replace("<asset>", "<asset>" + assets, 1)
    xml = re.sub(r'(<geom name="nominal_floor")', r'\1 material="sx_grid" group="2"', xml, count=1)
    xml = xml.replace("<worldbody>", '<worldbody><light pos="2 -2 4" dir="-0.4 0.4 -1" diffuse="0.8 0.8 0.8" '
                      'castshadow="true"/><light pos="-2 2 3" dir="0.4 -0.4 -1" diffuse="0.35 0.35 0.4" castshadow="false"/>', 1)
    xml = xml.replace("<mujoco ", '<mujoco ', 1)
    if "<visual>" not in xml:
        xml = xml.replace("</mujoco>", '<visual><global offwidth="1280" offheight="720"/><quality shadowsize="2048"/>'
                          '<headlight ambient="0.35 0.35 0.35" diffuse="0.3 0.3 0.3"/></visual></mujoco>')
    cwd = os.getcwd()
    os.chdir(scene.parent)  # mesh paths are relative to the scene
    try:
        m = mujoco.MjModel.from_xml_string(xml)
    finally:
        os.chdir(cwd)
    if tint is not None:
        robot = m.geom_bodyid > 0
        m.geom_rgba[robot, :3] = 0.55 * m.geom_rgba[robot, :3] + 0.45 * np.asarray(tint)
    return m


class View:
    def __init__(self, m):
        self.m, self.d = m, mujoco.MjData(m)
        self.r = mujoco.Renderer(m, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = [0, 1, 1, 0, 0, 0]  # visual meshes and the floor; collision geoms hidden
        self.look = None

    def frame(self, qpos, azimuth, elevation=-12.0, distance=2.6):
        self.d.qpos[:] = qpos
        mujoco.mj_forward(self.m, self.d)
        target = qpos[:3].copy()
        target[2] = max(0.65, float(qpos[2]) - 0.05)  # follow the root up in flight
        self.look = target if self.look is None else 0.9 * self.look + 0.1 * target  # smooth tracking
        self.cam.lookat[:], self.cam.azimuth, self.cam.elevation, self.cam.distance = self.look, azimuth, elevation, distance
        self.r.update_scene(self.d, self.cam, self.opt)
        return self.r.render()


def tracking_metrics(m, meta, qpos, ref_qpos, alive):
    d = mujoco.MjData(m)
    ids = np.asarray(meta["tracked_body_ids"])
    fq = meta["free_qpos_adr"]
    err_w, err_r = [], []
    for q, qr in zip(qpos[alive], ref_qpos[alive]):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        a = d.xpos[ids].copy()
        d.qpos[:] = qr
        mujoco.mj_kinematics(m, d)
        b = d.xpos[ids].copy()
        err_w.append(np.linalg.norm(a - b, axis=-1).mean())
        err_r.append(np.linalg.norm((a - q[fq : fq + 3]) - (b - qr[fq : fq + 3]), axis=-1).mean())
    fell = not bool(alive[-1])
    # joint speed against the rated (24 V) speed: the simulation limits torque only, so report what a speed limit
    # would have cut
    rated = {j["joint"]: j.get("rated_velocity") for j in meta.get("actuator_profile", {}).get("joints", [])}
    speed = np.abs(np.diff(qpos[alive][:, meta["qpos_adr"]], axis=0)) * FPS
    ratio = [float(speed[:, i].max() / rated[n]) for i, n in enumerate(meta["joint_names"]) if rated.get(n) and len(speed)]
    worst = int(np.argmax(ratio)) if ratio else None
    over = {"peak_speed_over_rated": max(ratio) if ratio else None,
            "peak_speed_joint": [n for n in meta["joint_names"] if rated.get(n)][worst] if ratio else None}
    return {"fell": fell, **over, "survived_frac": float(alive.mean()), "steps": int(len(alive) - 1),
            "body_pos_err_m": float(np.mean(err_w)), "body_pos_err_rootrel_m": float(np.mean(err_r))}


def card(lines, size=(2 * W, 2 * H)):
    img = Image.new("RGB", size, (10, 12, 18))
    dr = ImageDraw.Draw(img)
    y = size[1] // 2 - 90
    for text, pt, colour in lines:
        f = font(pt)
        w = dr.textlength(text, font=f)
        dr.text(((size[0] - w) / 2, y), text, font=f, fill=colour)
        y += pt + 22
    return np.asarray(img)


def label(frame, text, sub=None):
    img = Image.fromarray(frame)
    dr = ImageDraw.Draw(img)
    dr.rectangle((0, 0, W, 30), fill=(0, 0, 0))
    dr.text((10, 6), text, font=font(17), fill=(235, 235, 235))
    if sub:
        dr.text((W - 10 - dr.textlength(sub, font=font(15)), 8), sub, font=font(15), fill=(170, 190, 220))
    return np.asarray(img)


def write_video(path, clip, r, metrics, tagline, sim_m, ref_m):
    import imageio.v2 as imageio

    sim3, simS, ref3, refS = View(sim_m), View(sim_m), View(ref_m), View(ref_m)
    status = "fell" if metrics["fell"] else "tracked to the end"
    title = card([(clip.replace("_", " "), 46, (245, 245, 245)), (tagline, 24, (170, 190, 220)),
                  (f"{status} · body error {100 * metrics['body_pos_err_m']:.1f} cm", 22, (200, 200, 200))])
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=8) as w:
        for _ in range(2 * FPS):
            w.append_data(title)
        for t, (q, qr, ok) in enumerate(zip(r["qpos"], r["ref_qpos"], r["alive"])):
            sec = f"{t / FPS:5.2f} s"
            tag = "SIM" if ok else "SIM · FELL"
            a = label(sim3.frame(q, 135), f"{tag} · 3/4 view", sec)
            b = label(ref3.frame(qr, 135), "REFERENCE (kinematic)", clip[:28])
            c = label(simS.frame(q, 90), f"{tag} · side view")
            e = label(refS.frame(qr, 90), "REFERENCE · side view")
            w.append_data(np.concatenate([np.concatenate([a, b], 1), np.concatenate([c, e], 1)], 0))


def speed_note(r):
    peak = r.get("peak_speed_over_rated")
    return "" if peak is None else f" · peak joint speed {peak:.1f}× rated ({html.escape(r['peak_speed_joint'])})"


def speed_cell(r):
    peak = r.get("peak_speed_over_rated")
    return "" if peak is None else f"{peak:.2f}"


def gallery(out: Path, rows, header):
    cells = []
    for r in rows:
        badge = '<span class="bad">fell</span>' if r["fell"] else '<span class="ok">tracked</span>'
        cells.append(f'<figure><video src="{html.escape(r["clip"])}.mp4" controls loop muted preload="metadata"></video>'
                     f'<figcaption><b>{html.escape(r["clip"])}</b> {badge}<br>body error {100 * r["body_pos_err_m"]:.1f} cm · '
                     f'root-relative {100 * r["body_pos_err_rootrel_m"]:.1f} cm · survived {100 * r["survived_frac"]:.0f}%'
                     f'{speed_note(r)}'
                     f'</figcaption></figure>')
    table = "".join(f'<tr><td>{html.escape(r["clip"])}</td><td>{"fell" if r["fell"] else "no"}</td>'
                    f'<td>{100 * r["survived_frac"]:.0f}%</td><td>{100 * r["body_pos_err_m"]:.1f}</td>'
                    f'<td>{100 * r["body_pos_err_rootrel_m"]:.1f}</td><td>{r["steps"] / FPS:.1f}</td>'
                    f'<td>{speed_cell(r)}</td></tr>' for r in rows)
    ok = sum(not r["fell"] for r in rows)
    (out / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Sentient humanoid · XMimic</title>
<style>body{{background:#0b0d12;color:#e8e8ea;font:15px -apple-system,Helvetica,sans-serif;margin:32px}}
h1{{font-size:28px;margin:0 0 6px}} .sub{{color:#9fb3d6;margin-bottom:22px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:22px}}
video{{width:100%;border-radius:8px;background:#000}} figure{{margin:0}} figcaption{{margin-top:6px;color:#c8c8cc}}
.ok{{background:#1f7a4d;padding:1px 8px;border-radius:9px}} .bad{{background:#a3343a;padding:1px 8px;border-radius:9px}}
table{{border-collapse:collapse;margin:26px 0}} td,th{{padding:5px 14px;border-bottom:1px solid #262a33;text-align:left}}</style>
<h1>Sentient humanoid · motion tracking in simulation</h1>
<div class="sub">{html.escape(header)}<br>{ok} of {len(rows)} clips tracked to the end without falling.
Simulation only (MuJoCo); no physical robot was commanded. Actuators are torque-limited to their sx-embodiments
ratings; joint speed is not limited in simulation, so each clip reports its peak joint speed against the rated 24 V speed. Left: simulated robot under the policy. Right: the kinematic
reference motion it tracks.</div>
<div class="grid">{''.join(cells)}</div>
<table><tr><th>clip</th><th>fell</th><th>survived</th><th>body err (cm)</th><th>root-rel err (cm)</th><th>length (s)</th><th>peak speed / rated</th></tr>
{table}</table>""")


def evaluate_clip(job):
    """Roll out one clip from its first frame, score it and render its video (one process per clip with --jobs)."""
    ckpt, clip, motion_dir, robot_dir, out, tagline, no_video = job
    robot = Path(robot_dir)
    meta = json.loads((robot / "meta.json").read_text())
    phys = mujoco.MjModel.from_xml_path(str(robot / "scene.xml"))
    r = rollout(ckpt, clip, 0.0, None, motion_dir, robot_dir)
    metrics = {"clip": clip, **tracking_metrics(phys, meta, r["qpos"], r["ref_qpos"], np.asarray(r["alive"]))}
    print(json.dumps(metrics), flush=True)
    if not no_video:
        sim_m = render_model(robot / "scene.xml")
        ref_m = render_model(robot / "scene.xml", tint=(0.25, 0.5, 1.0))
        write_video(str(Path(out) / f"{clip}.mp4"), clip, r, metrics, tagline, sim_m, ref_m)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--robot_dir", required=True)
    ap.add_argument("--motion_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", default="", help="comma-separated; default: every clip")
    ap.add_argument("--label", default="datasheet actuators")
    ap.add_argument("--run", default="")
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="clips evaluated and rendered in parallel processes")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    robot = Path(a.robot_dir)
    clips = [c for c in a.clips.split(",") if c] or sorted(p.stem for p in Path(a.motion_dir).glob("*.npz"))
    render_meshes(robot / "scene.xml", (robot / "scene.xml").read_text())  # build the shared mesh cache once
    it = json.loads(str(np.load(a.ckpt)["meta"]))["iteration"]
    tagline = f"Simulation · Sentient humanoid · {a.label} · XMimic"
    jobs = [(a.ckpt, clip, a.motion_dir, str(robot), str(out), tagline, a.no_video) for clip in clips]
    if a.jobs > 1:
        import concurrent.futures
        import multiprocessing

        with concurrent.futures.ProcessPoolExecutor(a.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            rows = list(pool.map(evaluate_clip, jobs))
    else:
        rows = [evaluate_clip(j) for j in jobs]
    header = f"Policy {a.run} checkpoint iteration {it} · {a.label} · XMimic world-model tracker trained on Modal"
    summary = {"checkpoint": os.path.abspath(a.ckpt), "iteration": it, "run": a.run, "label": a.label, "clips": rows,
               "tracked": sum(not r["fell"] for r in rows), "total": len(rows)}
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    with (out / "metrics.csv").open("w") as f:
        f.write("clip,fell,survived_frac,body_pos_err_m,body_pos_err_rootrel_m,steps,peak_speed_over_rated,peak_speed_joint\n")
        for r in rows:
            f.write(f"{r['clip']},{int(r['fell'])},{r['survived_frac']:.4f},{r['body_pos_err_m']:.4f},"
                    f"{r['body_pos_err_rootrel_m']:.4f},{r['steps']},{r.get('peak_speed_over_rated')},{r.get('peak_speed_joint')}\n")
    gallery(out, rows, header)
    print(f"{summary['tracked']}/{summary['total']} tracked -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
