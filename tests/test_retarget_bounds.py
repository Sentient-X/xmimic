"""A clipped unconstrained step is not a constrained stationarity certificate."""

import numpy as np

from test_retarget_target import analytic, fixture_body
from xmimic.retarget_target import IKConfig, Landmark, TargetProfile, prepare_target, solve_sequence


def test_feasible_pose_is_reached_from_bound_where_clipped_newton_step_stalls(tmp_path, monkeypatch):
    body = fixture_body(tmp_path, monkeypatch)
    target = prepare_target(body, TargetProfile((Landmark('point', 'tip_site', orientation=True),),
                            holds=(('q3', .3),)), expected_embodiment_id=str(body.id))
    # Native order is q2,q1,q3. The old unconstrained DLS update clipped at
    # [2,-2,.3] rejects every backtrack, although a feasible descent remains.
    # Independent planar geometry supplies a reachable interior goal.
    point, rotation = analytic(.6, 1.1, 0., np.zeros(3))
    result = solve_sequence(target, {'point': point[None]}, orientations={'point': rotation[None]},
                            root_p=np.zeros((1, 3)), root_quat=np.array([[1., 0., 0., 0.]]),
                            fps=30., initial_q=[2., -2., .3],
                            config=IKConfig(position_tolerance_m=1e-6, orientation_tolerance_rad=1e-6))
    assert result.feasible
    np.testing.assert_allclose(result.q[0], [1.1, .6, .3], atol=1e-5)
    np.testing.assert_array_equal(result.q[:, 2], .3)
    assert np.all(result.q >= np.asarray(target.binding.lower))
    assert np.all(result.q <= np.asarray(target.binding.upper))
