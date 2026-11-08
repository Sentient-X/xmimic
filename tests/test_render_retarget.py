import numpy as np
import pytest
from tools.render_retarget import display_indices, limit_hits


def test_display_downsamples_only_and_keeps_both_endpoints():
    frames = display_indices(599, 1 / 0.008333, 30)
    assert len(frames) == 150
    assert frames[0] == 0 and frames[-1] == 598
    assert np.all(np.diff(frames) > 0)
    assert np.max(np.abs(frames[:-1] * 0.008333 - np.arange(149) / 30)) <= 0.008333 / 2


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_display_fps_refused(fps):
    with pytest.raises(ValueError):
        display_indices(599, 120, fps)


def test_limit_display_excludes_declared_holds_and_includes_lower_and_upper():
    q = np.array([[0.0, 1.0, 0.5], [-1.0, 0.0, 0.5]])
    np.testing.assert_array_equal(
        limit_hits(q, [-1, -1, 0.5], [1, 1, 0.5], [True, True, False]),
        [[False, True, False], [True, False, False]],
    )


def _angular_profile():
    from xmimic.retarget_target import Direction, Landmark, TargetProfile

    return TargetProfile(
        landmarks=(Landmark("Wrist", "hand", orientation=True, position=False),),
        directions=(Direction("Shoulder", "Elbow", "pivot", "elbow"),),
    )


def test_zero_unconstrained_positions_cannot_hide_wrong_bend_or_orientation():
    from tools.render_retarget import diagnostic_metrics
    from xmimic.retarget_target import IKConfig

    metrics = diagnostic_metrics(
        {
            "position_residual_m": [[0], [0], [0]],
            "orientation_residual_rad": [[0], [0.02], [0.001]],
            "direction_residual_rad": [[np.radians(20)], [0], [np.radians(2)]],
        },
        _angular_profile(),
        IKConfig(),
        3,
    )
    np.testing.assert_array_equal(metrics.goals_met, [False, False, True])
    np.testing.assert_allclose(metrics.curve, [20, 0, 2])
    assert metrics.curve_tolerance == pytest.approx(5)
    assert metrics.direction_keys == ("Shoulder->Elbow",)


def test_legacy_report_keeps_position_gate_and_ignores_unrequested_orientation():
    from tools.render_retarget import diagnostic_metrics
    from xmimic.retarget_target import Landmark, TargetProfile, IKConfig

    metrics = diagnostic_metrics(
        {
            "position_residual_m": [[0.001], [0.003]],
            "orientation_residual_rad": [[1], [1]],
        },
        TargetProfile((Landmark("Wrist", "hand"),)),
        IKConfig(),
        2,
    )
    np.testing.assert_array_equal(metrics.goals_met, [True, False])
    np.testing.assert_allclose(metrics.curve, [1, 3])
    assert metrics.direction_keys == ()
    assert "position" in metrics.curve_label


@pytest.mark.parametrize("direction", [None, [[0]], [[float("nan")], [0]], [[-1], [0]]])
def test_new_direction_reports_refuse_missing_partial_or_invalid_evidence(direction):
    from tools.render_retarget import diagnostic_metrics
    from xmimic.retarget_target import IKConfig

    report = {"position_residual_m": [[0], [0]], "orientation_residual_rad": [[0], [0]]}
    if direction is not None:
        report["direction_residual_rad"] = direction
    with pytest.raises(ValueError, match="Direction targets|direction_residual"):
        diagnostic_metrics(report, _angular_profile(), IKConfig(), 2)


def test_overlays_use_declared_link_origins_after_joint_motion():
    import mujoco
    from tools.render_retarget import segment_indices, target_point_position
    from xmimic.retarget_target import Direction, TargetProfile

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="distractor" pos="9 9 9"/>
      <body name="pivot" pos="1 2 3"><joint axis="0 0 1"/>
        <geom type="sphere" size=".1"/>
        <body name="elbow" pos="1 0 0"><body name="wrist" pos="1 0 0"/></body>
      </body></worldbody></mujoco>""")

    class Clip:
        def joint_index(self, name):
            return {"Elbow": 0, "Wrist": 1, "Shoulder": 2}[name]

    profile = TargetProfile(
        (),
        directions=(
            Direction("Shoulder", "Elbow", "pivot", "elbow"),
            Direction("Elbow", "Wrist", "elbow", "wrist"),
        ),
    )
    pairs = segment_indices(profile, Clip(), model)
    assert [pair[:2] for pair in pairs] == [(2, 0), (0, 1)]
    data = mujoco.MjData(model)
    data.qpos[0] = np.pi / 2
    mujoco.mj_kinematics(model, data)
    np.testing.assert_allclose(
        [target_point_position(data, p) for p in pairs[0][2:]],
        [[1, 2, 3], [1, 3, 3]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [target_point_position(data, p) for p in pairs[1][2:]],
        [[1, 3, 3], [1, 4, 3]],
        atol=1e-12,
    )


def test_site_overlay_uses_canonical_offsets_rotated_with_owning_link(
    tmp_path, monkeypatch
):
    import mujoco
    from dataclasses import replace
    from sx_contracts import ComponentId
    from sx_embodiments import Embodiment
    from sx_embodiments.assembly import Pose
    from sx_embodiments.sites import BodySite
    from test_retarget_target import fixture_body
    from tools.render_retarget import segment_indices, target_point_position
    from xmimic.retarget_target import Direction, TargetKind, TargetProfile

    body = replace(
        fixture_body(tmp_path, monkeypatch),
        sites=(
            BodySite(
                "shoulder_pivot",
                ComponentId("arm"),
                "a",
                Pose((0.2, 0.3, 0.0), (0.0, 0.0, 0.5)),
            ),
            BodySite(
                "elbow_pivot",
                ComponentId("arm"),
                "b",
                Pose((0.1, -0.2, 0.0), (0.0, 0.0, 0.0)),
            ),
        ),
    )
    body = Embodiment.from_json(
        body.to_json()
    )  # the same portable document boundary as render
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="a" pos="1 2 3"><joint axis="0 0 1"/><geom type="sphere" size=".1"/>
        <body name="b" pos="1 0 0"/>
      </body></worldbody></mujoco>""")
    assert model.nsite == 0

    class Clip:
        def joint_index(self, name):
            return {"Shoulder": 0, "Elbow": 1}[name]

    profile = TargetProfile(
        (),
        directions=(
            Direction(
                "Shoulder",
                "Elbow",
                "shoulder_pivot",
                "elbow_pivot",
                kind=TargetKind.SITE,
            ),
        ),
    )
    with pytest.raises(ValueError, match="canonical embodiment"):
        segment_indices(profile, Clip(), model)
    pairs = segment_indices(profile, Clip(), model, body)
    data = mujoco.MjData(model)
    data.qpos[0] = np.pi / 2
    mujoco.mj_kinematics(model, data)
    np.testing.assert_allclose(
        target_point_position(data, pairs[0][2]), [0.7, 2.2, 3], atol=1e-12
    )
    np.testing.assert_allclose(
        target_point_position(data, pairs[0][3]), [1.2, 3.1, 3], atol=1e-12
    )


def test_direction_diagnostics_include_unsampled_middle_frames(tmp_path, monkeypatch):
    from dataclasses import replace
    import mujoco
    from sx_contracts import ComponentId
    from sx_embodiments.sites import BodySite
    from sx_embodiments.assembly import Pose
    from test_retarget_target import fixture_body
    from tools.render_retarget import raw_direction_diagnostics, segment_indices
    from xmimic.retarget_target import Direction, TargetKind, TargetProfile

    body = replace(
        fixture_body(tmp_path, monkeypatch),
        sites=(
            BodySite("origin", ComponentId("arm"), "a"),
            BodySite("material_tip", ComponentId("arm"), "a", Pose((0.4, 0.0, 0.0))),
        ),
    )
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="base"><freejoint/><geom type="sphere" size=".1"/>
        <body name="a"><joint axis="0 0 1"/><geom type="sphere" size=".1"/></body>
      </body></worldbody></mujoco>""")

    class Clip:
        world_positions = np.array([[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]] * 3)

        def joint_index(self, name):
            return {"sensor": 0, "ray": 1}[name]

    profile = TargetProfile(
        (),
        directions=(
            Direction("sensor", "ray", "origin", "material_tip", kind=TargetKind.SITE),
        ),
    )
    clip = Clip()
    metrics = raw_direction_diagnostics(
        profile,
        clip,
        segment_indices(profile, clip, model, body),
        model,
        {"free_qpos_adr": 0, "qpos_adr": np.array([7])},
        np.zeros((3, 3)),
        np.array([[1.0, 0.0, 0.0, 0.0]] * 3),
        np.array([[0.0], [np.pi / 2], [0.0]]),
    )
    assert display_indices(3, 30, 1).tolist() == [0, 2]
    np.testing.assert_allclose(
        metrics["sensor->ray"]["angular_deg"], [0, 90, 0], atol=1e-10
    )
    np.testing.assert_allclose(
        metrics["sensor->ray"]["projected_heading_error_deg"], [0, 90, 0], atol=1e-10
    )


def test_heading_projection_is_gravity_relative_and_reports_undefined():
    from tools.render_retarget import direction_errors

    angular, heading = direction_errors(
        [1, 1, 0], [1, -1, 0], np.array([0.0, 1.0, 0.0])
    )
    assert angular == pytest.approx(90) and heading == pytest.approx(0)
    angular, heading = direction_errors([0, 0, 1], [1, 0, 0], np.array([0.0, 0.0, 1.0]))
    assert angular == pytest.approx(90) and heading is None
    assert direction_errors([0, 0, 0], [1, 0, 0], np.array([0.0, 0.0, 1.0])) == (
        None,
        None,
    )
    assert direction_errors([1, 0, 0], [0, 1, 0], None) == (90.0, None)
