"""Differentiable scalar-tree FK checked against independently compiled MuJoCo."""
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from xmimic.kin import Kinematics
from xmimic.rot import so3_log


def mechanism():
    model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
      <worldbody><body name="mount" pos=".3 -.2 .5" euler=".2 -.4 .3">
        <body name="carrier" pos=".1 .2 -.1" euler="-.3 .2 .1">
          <joint name="turn" type="hinge" axis="0 1 0" pos=".03 -.02 .04" ref=".12" range="-2 2"/>
          <joint name="slide" type="slide" axis=".6 .8 0" ref=".02" range="-.2 .2"/>
          <geom size=".02" mass=".2"/>
          <body name="mirror_node" pos=".1 -.2 .3" euler=".1 .2 -.3">
            <joint name="mirror" axis="1 0 0" pos=".02 .03 -.04" range="-2 2"/>
            <geom size=".02" mass=".1"/>
            <body name="follower_node" pos=".2 .1 .05"><joint name="follower" axis="0 0 1" range="-2 2"/>
              <geom size=".02" mass=".1"/>
              <body name="fixed_tip" pos="-.1 .2 .3" euler=".3 .2 .1"/>
            </body></body></body></body></worldbody></mujoco>''')
    dependencies = ({'joint': 'mirror', 'leader': 'turn', 'multiplier': -.4, 'offset': .1},
                    {'joint': 'follower', 'leader': 'mirror', 'multiplier': 1.3, 'offset': -.2})
    return model, dependencies


def independent(model, q):
    data = mujoco.MjData(model)
    angles = {'slide': q[0], 'turn': q[1], 'mirror': -.4*q[1]+.1, 'follower': 1.3*(-.4*q[1]+.1)-.2}
    for name, value in angles.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    mujoco.mj_forward(model, data)
    return data.xpos.copy(), data.xmat.reshape(-1, 3, 3).copy()


def test_native_permutation_prismatic_mimic_and_offset_hinges_match_mujoco():
    model, dependencies = mechanism()
    kin = Kinematics.from_mujoco(model, ('slide', 'turn'), dependent_joints=dependencies)
    q = jnp.asarray([[.02, .12], [.11, -.7], [-.15, .8]])
    zero = jnp.zeros((3, 3))
    positions, rotations, _, _ = kin.forward(zero, jnp.tile(jnp.eye(3), (3, 1, 1)), zero, zero, q, jnp.zeros_like(q))
    expected = [independent(model, row) for row in np.asarray(q)]
    np.testing.assert_allclose(positions, np.asarray([p for p, _ in expected]), atol=2e-7)
    np.testing.assert_allclose(rotations, np.asarray([R for _, R in expected]), atol=2e-7)


def test_jacobians_and_velocities_include_anchor_and_mimic_scale():
    model, dependencies = mechanism()
    kin = Kinematics.from_mujoco(model, ('slide', 'turn'), dependent_joints=dependencies)
    q, qd = np.array([.11, -.7]), np.array([.07, -.3])
    def forward(values):
        return kin.forward(jnp.zeros(3), jnp.eye(3), jnp.zeros(3), jnp.zeros(3), values, jnp.asarray(qd))
    actual = forward(jnp.asarray(q))
    jacobian = jax.jacrev(lambda values: forward(values)[:2])(jnp.asarray(q))
    eps = 1e-5
    differences = [(independent(model, q+delta), independent(model, q-delta)) for delta in np.eye(2)*eps]
    jp = np.stack([(a[0]-b[0])/(2*eps) for a, b in differences], axis=-1)
    jr = np.stack([(a[1]-b[1])/(2*eps) for a, b in differences], axis=-1)
    np.testing.assert_allclose(jacobian[0], jp, atol=3e-7)
    np.testing.assert_allclose(jacobian[1], jr, atol=3e-7)
    np.testing.assert_allclose(actual[2], jp@qd, atol=3e-7)
    _, rotation = independent(model, q)
    skew = (jr@qd)@np.swapaxes(rotation, -1, -2)
    expected_angular = np.stack((skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]), axis=-1)
    np.testing.assert_allclose(actual[3], expected_angular, atol=3e-7)
    zero_jacobian = jax.jacrev(lambda values: forward(values)[:2])(jnp.zeros(2))
    assert all(np.isfinite(value).all() for value in zero_jacobian)


def test_unsupported_or_unbound_coordinates_fail_explicitly():
    model, dependencies = mechanism()
    with pytest.raises(ValueError, match='every moving coordinate'):
        Kinematics.from_mujoco(model, ('slide', 'turn'))
    bad = ({'joint': 'mirror', 'leader': 'follower', 'multiplier': 1., 'offset': 0.}, dependencies[1])
    with pytest.raises(ValueError, match='cyclic'):
        Kinematics.from_mujoco(model, ('slide', 'turn'), dependent_joints=bad)
    ball = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="b" type="ball"/><geom size=".1"/></body></worldbody></mujoco>')
    with pytest.raises(ValueError, match='free or ball'):
        Kinematics.from_mujoco(ball, ('b',))


def test_safe_rotation_log_has_finite_identity_derivative_and_same_values():
    from xmimic.rot import axis_angle_to_mat
    def log(angle, safe):
        return so3_log(axis_angle_to_mat(jnp.array([0., 1., 0.]), angle), safe_zero=safe)
    np.testing.assert_allclose(jax.jacrev(lambda angle: log(angle, True))(jnp.array(0.)), [0., 1., 0.], atol=1e-7)
    for angle in (-2., -.01, 0., .3, 3.1):
        np.testing.assert_array_equal(log(jnp.array(angle), True), log(jnp.array(angle), False))
