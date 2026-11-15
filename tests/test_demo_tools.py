"""CPU checks of the Modal demo tools: the datasheet gain rule, the render-mesh decimation and the cost ledger."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import datasheet_profile as dp  # noqa: E402


def test_ratings_cover_the_sentient_joints_and_follow_the_gain_rule():
    names = ["torso_joint", "waist_roll_joint", "left_hip_pitch_joint", "right_knee_joint", "left_foot2_joint",
             "right_hip_yaw_joint", "left_elbow_pitch_joint", "right_wrist_roll_joint", "left_gripper_left_finger_joint"]
    efforts = {n: dp.rating(n)[1] for n in names}
    assert efforts["left_hip_pitch_joint"] == efforts["right_knee_joint"] == 120.0
    assert efforts["right_hip_yaw_joint"] == efforts["torso_joint"] == 40.0
    assert efforts["left_foot2_joint"] == efforts["waist_roll_joint"] == 27.0
    with pytest.raises(ValueError):
        dp.rating("neck_yaw_joint")
    armature = dp.rating("left_knee_joint")[2]
    assert math.isclose(armature * dp.W**2, armature * (2 * math.pi * 10) ** 2)


def test_cluster_mesh_welds_and_drops_collapsed_faces():
    pytest.importorskip("PIL")
    import render_demo as rd

    rng = np.random.default_rng(0)
    v = rng.uniform(0, 0.1, (3000, 3))
    f = rng.integers(0, 3000, (6000, 3))
    w, g = rd.cluster_mesh(v, f, 0.02)
    assert len(w) <= 125 and len(g) < len(f)
    assert (g[:, 0] != g[:, 1]).all() and (g[:, 1] != g[:, 2]).all() and (g[:, 0] != g[:, 2]).all()
    assert len(np.unique(g, axis=0)) == len(g)


def test_ledger_accumulates(tmp_path):
    modal = pytest.importorskip("modal")  # noqa: F841
    import modal_app

    path = tmp_path / "costs.json"
    assert modal_app._book(path, {"usd": 0.5}) == 0.5
    assert modal_app._book(path, {"usd": 0.25}) == 0.75
    assert len(json.loads(path.read_text())["calls"]) == 2
