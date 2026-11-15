"""Render the judging set for retargeted clips: phase keyframes in four views, a metrics summary and a
real-time side-by-side video. Pictures are made to be read by a person or an agent at view size.

    python -m tools.judge_retarget --body-document BODY.json --scene SCENE.xml \
        --calibration-source CAL.bvh --correspondence tools/soma_correspondence.json RUN_DIR/<clip> ...

Writes <clip>/judge/{keyframes.png, keyframes_overlay.png, summary.png, sidebyside.mp4, keyframes.json}.
Colours: left limbs warm, right limbs cool, trunk neutral; every robot link takes its segment's colour
and the scaled human skeleton (the solve's targets) is drawn with the same colours.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from tools.retarget_clips import build  # noqa: E402

WARM = [(0.95, 0.25, 0.15), (1.0, 0.6, 0.0), (0.95, 0.85, 0.1), (0.85, 0.2, 0.55), (0.7, 0.35, 0.1)]
COOL = [(0.1, 0.45, 0.95), (0.0, 0.8, 0.85), (0.2, 0.8, 0.3), (0.45, 0.3, 0.9), (0.1, 0.3, 0.5)]
TRUNK = (0.75, 0.75, 0.78)


def segment_colours(r) -> dict[tuple[str, str], tuple]:
    """Side of each segment from its child site's lateral position in the calibration pose."""
    m = r.model
    d = mujoco.MjData(m)
    d.qpos[:] = r.calibration_qpos
    mujoco.mj_kinematics(m, d)
    root_p, root_R = r._site_pose(d, r.root_site)
    sites = {n: root_R.T @ (r._site_pose(d, n)[0] - root_p) for n in r.sites}
    centre = np.mean([v[1] for v in sites.values()])
    out, w, c = {}, 0, 0
    for s in r.segments:
        y = sites[s.child][1] - centre
        if abs(y) < 0.03 and abs(sites[s.parent][1] - centre) < 0.03:
            out[(s.parent, s.child)] = TRUNK
        elif y > 0:
            out[(s.parent, s.child)] = WARM[w % len(WARM)]
            w += 1
        else:
            out[(s.parent, s.child)] = COOL[c % len(COOL)]
            c += 1
    return out


def body_colours(r, colours) -> np.ndarray:
    """RGB per body: bodies on the path parent-site link .. child-site link take the segment colour."""
    m = r.model
    col = np.full((m.nbody, 3), np.nan)

    def path(a: int, b: int) -> list[int]:
        out = []
        while b != a and b > 0:
            out.append(b)
            b = m.body_parentid[b]
        return out[::-1]

    leaves = {s.child for s in r.segments} - {s.parent for s in r.segments}
    col[r.sites[r.root_site].body] = TRUNK
    for s in r.segments:
        a, b = r.sites[s.parent].body, r.sites[s.child].body
        rgb = colours[(s.parent, s.child)]
        chain = [a] + path(a, b)
        if s.child not in leaves:
            chain = chain[:-1]
        for body in chain[1:] if np.isfinite(col[a]).all() else chain:
            col[body] = rgb
        if s.child in leaves:
            col[b] = rgb
    for body in range(1, m.nbody):  # remaining bodies inherit the nearest coloured ancestor
        if not np.isfinite(col[body]).all():
            p = body
            while p > 0 and not np.isfinite(col[p]).all():
                p = m.body_parentid[p]
            col[body] = col[p] if p > 0 else TRUNK
    return col


class Scene:
    def __init__(self, r, width: int, height: int):
        self.r = r
        m = r.model
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
        m.vis.global_.offheight = max(m.vis.global_.offheight, height)
        m.vis.quality.shadowsize = 2048
        m.vis.headlight.ambient[:] = 0.45
        m.vis.headlight.diffuse[:] = 0.65
        m.vis.headlight.specular[:] = 0.1
        floor = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0]
        m.geom_rgba[floor] = (0.82, 0.83, 0.86, 1.0)
        self.renderer = mujoco.Renderer(m, height, width)
        self.data = mujoco.MjData(m)
        self.colours = segment_colours(r)
        self.body_rgb = body_colours(r, self.colours)
        self.base_rgba = m.geom_rgba.copy()
        self.names = list(r.sites)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[1] = 1  # visual meshes only
        self.opt.geomgroup[0] = 1  # floor
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if b > 0:
                m.geom_rgba[g, :3] = self.body_rgb[b] * 0.85 + 0.1

    def set_alpha(self, alpha: float):
        m = self.r.model
        for g in range(m.ngeom):
            if m.geom_bodyid[g] > 0:
                m.geom_rgba[g, 3] = alpha

    def _capsule(self, a, b, radius, rgba):
        scn = self.renderer.scene
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, np.asarray(a, float), np.asarray(b, float))
        scn.ngeom += 1

    def _sphere(self, p, radius, rgba):
        scn = self.renderer.scene
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]), np.asarray(p, float),
                            np.eye(3).ravel(), np.asarray(rgba, np.float32))
        scn.ngeom += 1

    def render(self, qpos, targets, cam: dict, skeleton_shift=np.zeros(3), trace=None, contacts=(),
               skeleton_radius=0.022, grid=True):
        m, d = self.r.model, self.data
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = cam["lookat"]
        camera.distance, camera.azimuth, camera.elevation = cam["distance"], cam["azimuth"], cam["elevation"]
        self.renderer.update_scene(d, camera, self.opt)
        idx = {n: i for i, n in enumerate(self.names)}
        if grid:
            c = np.round(np.asarray(cam["lookat"][:2]) * 2) / 2
            for k in np.arange(-3, 3.01, 0.5):
                self._capsule([c[0] + k, c[1] - 3, 0.001], [c[0] + k, c[1] + 3, 0.001], 0.003, (0.3, 0.3, 0.35, 1))
                self._capsule([c[0] - 3, c[1] + k, 0.001], [c[0] + 3, c[1] + k, 0.001], 0.003, (0.3, 0.3, 0.35, 1))
        if trace is not None and len(trace) > 1:
            for a, b in zip(trace[:-1], trace[1:]):
                self._capsule(a, b, 0.006, (1, 1, 1, 0.9))
        for p in contacts:
            self._sphere(p, 0.03, (1, 0.1, 0.1, 0.9))
        if targets is not None:
            for s in self.r.segments:
                rgb = self.colours[(s.parent, s.child)]
                a = targets[idx[s.parent]] + skeleton_shift
                b = targets[idx[s.child]] + skeleton_shift
                self._capsule(a, b, skeleton_radius, (*rgb, 1.0))
            for n in self.names:
                self._sphere(targets[idx[n]] + skeleton_shift, skeleton_radius * 1.25, (0.1, 0.1, 0.1, 1))
        return self.renderer.render().copy()


def _label(img: np.ndarray, text: str, size: int = 22) -> np.ndarray:
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    dr.rectangle([0, 0, im.width, size + 10], fill=(0, 0, 0))
    dr.text((8, 4), text, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def phase_keyframes(q: np.ndarray, low: np.ndarray, sites: np.ndarray, names: list[str], fps: float) -> dict:
    """Keyframes by phase: takeoff, apex, most inverted, landing; else limb extremes."""
    up = Rotation.from_quat(q[:, 3:7], scalar_first=True).apply([0, 0, 1])
    tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
    air = low > 0.04
    keys: dict[str, int] = {}
    if air.any():
        first = int(np.argmax(air))
        keys["takeoff"] = max(first - 1, 0)
        apex_range = np.where(air)[0]
        keys["apex"] = int(apex_range[np.argmax(q[apex_range, 2])])
        after = np.where(~air[keys["apex"]:])[0]
        if len(after):
            keys["landing"] = int(keys["apex"] + after[0])
    if tilt.max() > 60:
        keys["most_inverted"] = int(np.argmax(tilt))
    ix = {n: i for i, n in enumerate(names)}
    leaves = [n for n in names if n.endswith(("hand", "toe_edge"))] or names
    for n in leaves:
        z = sites[:, ix[n]] [:, 2] - q[:, 2]
        keys.setdefault(f"{n}_highest", int(np.argmax(z)))
    keys.setdefault("deepest_crouch", int(np.argmin(q[:, 2])))
    ordered = dict(sorted(keys.items(), key=lambda kv: kv[1]))
    # keep at most 6, prefer phase keys
    prefer = [k for k in ("takeoff", "apex", "most_inverted", "landing") if k in ordered]
    rest = [k for k in ordered if k not in prefer]
    chosen = prefer + rest[: max(0, 6 - len(prefer))]
    return {k: ordered[k] for k in sorted(chosen, key=lambda k: ordered[k])}


def views(lookat, heading_deg: float) -> list[tuple[str, dict]]:
    la = np.asarray(lookat, float)
    return [
        ("front", {"lookat": la, "distance": 3.2, "azimuth": heading_deg + 180, "elevation": -8}),
        ("side", {"lookat": la, "distance": 3.2, "azimuth": heading_deg + 90, "elevation": -8}),
        ("top", {"lookat": la, "distance": 3.6, "azimuth": heading_deg + 180, "elevation": -88}),
        ("3/4", {"lookat": la, "distance": 3.0, "azimuth": heading_deg + 135, "elevation": -20}),
    ]


def judge_clip(r, clip_dir: Path, panel: int = 420, video_fps: int = 30) -> dict:
    sol = np.load(clip_dir / "solution.npz")
    ser = np.load(clip_dir / "series.npz")
    metrics = json.loads((clip_dir / "metrics.json").read_text())
    q, tp = sol["qpos"], sol["target_pos"]
    names = [str(x) for x in sol["site_names"]]
    fps = float(metrics["fps"])
    out = clip_dir / "judge"
    out.mkdir(exist_ok=True)
    scene = Scene(r, panel, panel)
    m = r.model
    d = mujoco.MjData(m)
    sites = np.empty((len(q), len(names), 3))
    for t in range(len(q)):
        d.qpos[:] = q[t]
        mujoco.mj_kinematics(m, d)
        sites[t] = [r._site_pose(d, n)[0] for n in names]
    keys = phase_keyframes(q, ser["low"], sites, names, fps)
    yaw0 = np.degrees(Rotation.from_quat(q[0, 3:7], scalar_first=True).as_euler("ZYX")[0])
    root_trace = q[::4, :3].copy()
    root_trace[:, 2] = 0.002
    seg_err = ser["seg_err"]
    rows_side, rows_over = [], []
    for name, t in keys.items():
        lookat = np.array([q[t, 0], q[t, 1], max(0.55, q[t, 2] * 0.9)])
        side_row, over_row = [], []
        for vname, cam in views(lookat, yaw0):
            az = np.radians(cam["azimuth"])
            right = np.array([np.sin(az), -np.cos(az), 0.0])  # camera right in the ground plane
            scene.set_alpha(1.0)
            img = scene.render(q[t], tp[t], cam, skeleton_shift=right * 0.95 if vname != "top" else
                               np.array([0, 0, 0]) + right * 0.95, trace=root_trace)
            side_row.append(_label(img, f"{name} f{t} {t / fps:.2f}s  {vname}  dir {seg_err[t].max():.0f}°"))
            scene.set_alpha(0.35)
            img = scene.render(q[t], tp[t], cam, skeleton_radius=0.014, trace=None)
            over_row.append(_label(img, f"{name} f{t} {vname} overlay"))
        rows_side.append(np.concatenate(side_row, 1))
        rows_over.append(np.concatenate(over_row, 1))
    scene.set_alpha(1.0)
    Image.fromarray(np.concatenate(rows_side, 0)).save(out / "keyframes.png")
    Image.fromarray(np.concatenate(rows_over, 0)).save(out / "keyframes_overlay.png")
    (out / "keyframes.json").write_text(json.dumps(keys))
    summary(metrics, ser, sol, fps, out / "summary.png", keys)
    video(r, q, tp, fps, out / "sidebyside.mp4", video_fps)
    return keys


def summary(metrics, ser, sol, fps, path: Path, keys):
    seg_err = ser["seg_err"]
    t = np.arange(len(seg_err)) / fps
    jn = [str(x) for x in ser["joint_names"]]
    fig, ax = plt.subplots(3, 2, figsize=(18, 13), constrained_layout=True)
    im = ax[0, 0].imshow(seg_err.T, aspect="auto", cmap="magma", vmin=0, vmax=45,
                         extent=[0, t[-1], len(metrics["segments"]) - 0.5, -0.5])
    ax[0, 0].set_yticks(range(len(metrics["segments"])), metrics["segments"], fontsize=9)
    ax[0, 0].set_title(f"segment direction error vs scaled human (deg)  p99 {np.percentile(seg_err, 99):.1f}")
    fig.colorbar(im, ax=ax[0, 0])
    sat = ser["sat"].astype(float)
    ax[0, 1].imshow(sat.T, aspect="auto", cmap="Reds", vmin=0, vmax=1, extent=[0, t[-1], len(jn) - 0.5, -0.5],
                    interpolation="nearest")
    ax[0, 1].set_yticks(range(len(jn)), [n.replace("_joint", "") for n in jn], fontsize=7)
    ax[0, 1].set_title("joint at limit (within 1 deg)")
    qd = np.abs(ser["qd"])
    order = np.argsort(-qd.max(0))[:10]
    for k in order:
        ax[1, 0].plot(t, qd[:, k], lw=1.2, label=f"{jn[k].replace('_joint', '')} {qd[:, k].max():.1f}")
    ax[1, 0].set_title("joint speed |rad/s| (10 fastest)")
    ax[1, 0].legend(fontsize=8, ncol=2)
    tau = np.abs(ser["tau"])
    order = np.argsort(-tau.max(0))[:10]
    for k in order:
        ax[1, 1].plot(t, tau[:, k], lw=1.2, label=f"{jn[k].replace('_joint', '')} {tau[:, k].max():.0f}")
    ax[1, 1].set_title("inverse-dynamics torque proxy |Nm| (10 largest)")
    ax[1, 1].legend(fontsize=8, ncol=2)
    q = sol["qpos"]
    ax[2, 0].plot(t, q[:, 2], label="robot root z (m)", color="tab:blue")
    ax[2, 0].plot(t, ser["low"], label="lowest body point (m)", color="tab:red")
    ax[2, 0].plot(t, sol["ground_shift"], label="ground shift (m)", color="tab:gray", ls="--")
    ax2 = ax[2, 0].twinx()
    ax2.plot(t, ser["tilt"], color="tab:purple", label="root tilt (deg)")
    ax2.set_ylim(0, 185)
    ax[2, 0].set_title("root height / lowest point / tilt")
    ax[2, 0].legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    for k, f in keys.items():
        ax[2, 0].axvline(f / fps, color="k", lw=0.6)
        ax[2, 0].text(f / fps, ax[2, 0].get_ylim()[1] * 0.95, k, rotation=90, fontsize=7, va="top")
    ax[2, 1].plot(t, ser["root_force"], label="root external force (body weights)")
    ax[2, 1].plot(t, ser["close"].astype(float) * 0.5, label="self proximity <4cm (flag)")
    hf = ser["hf"]
    ax[2, 1].set_title(f"physical plausibility; max HF jitter {hf.max():.2f} deg ({jn[int(np.argmax(hf))]})")
    ax[2, 1].legend(fontsize=8)
    sv = metrics["screen"]["values"]
    fig.suptitle(f"{metrics['clip']}  screen {'PASS' if metrics['screen']['passed'] else 'FAIL'}  "
                 + "  ".join(f"{k}={v:.3g}" for k, v in sv.items()), fontsize=12)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def video(r, q, tp, fps, path: Path, video_fps: int):
    import imageio.v2 as imageio

    scene = Scene(r, 960, 540)
    step = max(1, int(round(fps / video_fps)))
    yaw0 = np.degrees(Rotation.from_quat(q[0, 3:7], scalar_first=True).as_euler("ZYX")[0])
    writer = imageio.get_writer(path, fps=fps / step, codec="libx264", quality=7, macro_block_size=1)
    smooth = q[:, :3].copy()
    for t in range(0, len(q), step):
        la = smooth[max(0, t - 30): t + 30].mean(0)
        la[2] = 0.65
        cam = {"lookat": la, "distance": 3.4, "azimuth": yaw0 + 135, "elevation": -15}
        az = np.radians(cam["azimuth"])
        right = np.array([np.sin(az), -np.cos(az), 0.0])
        img = scene.render(q[t], tp[t], cam, skeleton_shift=right * 1.0)
        writer.append_data(_label(img, f"t={t / fps:5.2f}s  real time   robot | scaled human", 20))
    writer.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--body-document", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--calibration-source", required=True)
    p.add_argument("--calibration-frame", type=int, default=0)
    p.add_argument("--correspondence", required=True)
    p.add_argument("--retarget-limits", default=None)
    p.add_argument("clip_dirs", nargs="+")
    args = p.parse_args()
    r, _, _ = build(args)
    for c in args.clip_dirs:
        print(c, judge_clip(r, Path(c)), flush=True)


if __name__ == "__main__":
    main()
