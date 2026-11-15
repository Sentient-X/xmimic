"""The flight step makes a kinematic jump ballistic with constant angular momentum, for any body."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("scipy")

from xmimic.retarget_flight import angmom_drift, ballistic_residual, com_and_angmom, make_flights_consistent  # noqa: E402

XML = """
<mujoco><compiler angle="radian"/><worldbody><geom type="plane" size="5 5 .1"/>
<body name="torso" pos="0 0 1"><freejoint/><geom type="box" size=".1 .15 .2" mass="10"/>
  <body name="thigh" pos="0 0 -.2"><joint name="hip" axis="0 1 0" range="-2 2"/>
    <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".05" mass="3"/>
    <body name="foot" pos="0 0 -.45"><geom type="sphere" size=".03" mass=".5"/></body>
  </body>
</body></worldbody></mujoco>
"""


def _jump(m):
    fps, n = 120.0, 240
    t = np.arange(n) / fps
    q = np.tile(m.qpos0, (n, 1))
    lift = np.where((t > 0.6) & (t < 1.3), 0.4 * np.sin(np.pi * (t - 0.6) / 0.7), 0.0)  # not a parabola
    q[:, 2] = 0.65 + lift
    q[:, 0] = 0.5 * t
    pitch = np.where(t > 0.6, np.clip((t - 0.6) / 0.7, 0, 1), 0) * 2 * np.pi  # one full turn in the air (a flip)
    q[:, 3], q[:, 5] = np.cos(pitch / 2), np.sin(pitch / 2)
    q[:, 7] = 0.6 * np.sin(4 * (t - 0.6)) * (lift > 0)  # the leg swings in the air
    return q, fps


def _run(m, q, fps, speed):
    foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot")
    # lowest body point: the foot sphere centre (on the floor at torso height 0.65 m, leg straight)
    return make_flights_consistent(m, q, fps, lambda d: float(d.xpos[foot][2]), np.full(m.nv, speed),
                                   [(foot, np.zeros(3))])


def test_jump_becomes_ballistic_and_lands_as_referenced():
    m = mujoco.MjModel.from_xml_string(XML)
    q, fps = _jump(m)
    out, reports, tm = _run(m, q, fps, speed=200.0)
    assert len(reports) == 1
    rep = reports[0]
    assert rep.time_warp == 1.0 and rep.com_residual_before_m > 0.02
    s, e = rep.start, rep.new_end
    com, L = com_and_angmom(m, out[s + 8:e - 7], 1 / fps)
    assert ballistic_residual(com, 1 / fps) < 2e-3
    assert angmom_drift(L) < 0.05
    assert rep.landing_orientation_error_deg < 1.0
    np.testing.assert_allclose(out[:, 7], q[:, 7])  # joints untouched
    assert len(tm) == len(out)


def test_unreachable_takeoff_lowers_the_apex_and_shortens_the_flight():
    m = mujoco.MjModel.from_xml_string(XML)
    q, fps = _jump(m)
    out, reports, tm = _run(m, q, fps, speed=8.0)  # slow hip: little vertical takeoff speed
    rep = reports[0]
    assert rep.takeoff_vz_needed > rep.takeoff_vz_achievable
    assert rep.time_warp < 1.0
    # the new flight duration is the ballistic time of the achievable vertical speed
    assert rep.new_duration_s == pytest.approx(2 * rep.takeoff_vz_achievable / 9.81, abs=2 / fps)
    assert len(out) < len(q) and rep.landing_orientation_error_deg < 1.0
