"""Whole-body site-calibrated retargeting on a synthetic body whose own motion is the source.

The source skeleton is the body itself (source joint frames = site frames), so a correct
retargeter must reproduce the motion: site positions to millimetres, segment directions to
fractions of a degree, and the ground pass must keep the lowest point on the floor.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("scipy")

from xmimic.retarget_source import Clip  # noqa: E402
from xmimic.retarget_whole_body import WholeBodyRetargeter  # noqa: E402
from xmimic.retarget_judge import motion_metrics  # noqa: E402

XML = """
<mujoco><compiler angle="radian"/><worldbody><geom name="floor" type="plane" size="5 5 .1"/>
<body name="torso" pos="0 0 1"><freejoint/>
  <geom type="box" size=".1 .15 .2"/>
  <body name="l_hip" pos="0 .15 0"><joint name="l_hip_x" axis="1 0 0" range="-1 1"/><geom type="sphere" size=".03"/>
  <body name="l_upper"><joint name="l_hip_y" axis="0 1 0" range="-1.5 1.5"/>
    <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".04"/>
    <body name="l_lower" pos="0 0 -.4"><joint name="l_knee" axis="0 1 0" range="0 2.2"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".04"/>
      <body name="l_foot" pos="0 0 -.4"><geom type="box" pos=".05 0 -.02" size=".08 .04 .02"/></body>
    </body>
  </body></body>
  <body name="r_hip" pos="0 -.15 0"><joint name="r_hip_x" axis="1 0 0" range="-1 1"/><geom type="sphere" size=".03"/>
  <body name="r_upper"><joint name="r_hip_y" axis="0 1 0" range="-1.5 1.5"/>
    <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".04"/>
    <body name="r_lower" pos="0 0 -.4"><joint name="r_knee" axis="0 1 0" range="0 2.2"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".04"/>
      <body name="r_foot" pos="0 0 -.4"><geom type="box" pos=".05 0 -.02" size=".08 .04 .02"/></body>
    </body>
  </body></body>
</body></worldbody></mujoco>
"""
SITES = [("pelvis", "torso"), ("l_hip", "l_hip"), ("l_knee", "l_lower"), ("l_ankle", "l_foot"),
         ("r_hip", "r_hip"), ("r_knee", "r_lower"), ("r_ankle", "r_foot")]


def _mesh_free_lowest(r, data):
    # primitive geoms only: the lowest point is taken at the ankle sites, as the source's lowest joints are
    return min(data.xpos[r.sites[n].body][2] for n in ("l_ankle", "r_ankle"))


def _source_from_body(model, qs):
    d = mujoco.MjData(model)
    names = ["Root"] + [s for s, _ in SITES]
    parents = np.array([-1, 0, 1, 2, 3, 1, 5, 6])
    pos = np.zeros((len(qs), len(names), 3))
    rot = np.tile(np.eye(3), (len(qs), len(names), 1, 1))
    for t, q in enumerate(qs):
        d.qpos[:] = q
        mujoco.mj_kinematics(model, d)
        pos[t, 0] = [d.xpos[1][0], d.xpos[1][1], 0.0]
        for i, (_, body) in enumerate(SITES, start=1):
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            pos[t, i] = d.xpos[b]
            # a human bone frame follows its limb: the hip joint frame turns with the thigh
            frame = body.replace("_hip", "_upper") if body.endswith("_hip") else body
            rot[t, i] = d.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame)].reshape(3, 3)
    return Clip(tuple(names), parents, (("Xposition",),) * len(names), np.zeros((len(names), 3)), pos, rot,
                1 / 120, "0.008333", "synthetic", np.eye(4))


def test_self_retarget_reproduces_motion():
    model = mujoco.MjModel.from_xml_string(XML)
    doc = {"sites": [{"name": s, "link": b, "xyz": [0, 0, 0], "rpy": [0, 0, 0]} for s, b in SITES]}
    corr = {s: s for s, _ in SITES}
    t = np.arange(240) / 120
    qs = np.tile(model.qpos0, (len(t), 1))
    qs[:, 0] = 0.3 * t
    qs[:, 2] = 0.84 + 0.05 * np.sin(4 * t)
    for name, amp in (("l_hip_y", -0.6), ("l_knee", 1.0), ("r_hip_y", 0.4), ("r_knee", 0.6), ("l_hip_x", 0.2)):
        qs[:, model.jnt_qposadr[model.joint(name).id]] = amp * (0.5 - 0.5 * np.cos(3 * t))
    clip = _source_from_body(model, qs)
    r = WholeBodyRetargeter.from_document(model, doc, corr)
    r._lowest_point = lambda data: _mesh_free_lowest(r, data)
    report = r.calibrate(clip, 0)
    assert report["max_direction_deg"] < 0.5
    assert report["root_scale"] == pytest.approx(1.0, abs=1e-3)
    assert {s["child"]: s["rigid"] for s in report["segments"]} == {
        "l_hip": True, "r_hip": True, "l_knee": False, "r_knee": False, "l_ankle": True, "r_ankle": True}
    sol = r.solve(clip)
    assert sol.position_error_m.max() < 2e-3
    q = r.ground_and_smooth(clip, sol, clip.fps)
    hip0 = clip.world_positions[0, 1, :2]
    tp = np.asarray([[r.targets(clip, f, hip0)[n][0] for n in sol.site_names] for f in range(clip.frame_count)])
    tp[:, :, 2] += r.ground_shift[:, None]
    metrics = motion_metrics(r, q, tp, sol.site_names, clip.fps)
    assert metrics["screen"]["values"]["direction_max_deg"] < 1.0
    # the source's own ground contact is preserved, never pushed through the floor
    assert np.abs(r.ground_shift).max() < 5e-3
