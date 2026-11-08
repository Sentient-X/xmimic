"""Differentiable tree FK, retaining the original XMimic metadata/arithmetic path."""

import jax.numpy as jnp
import numpy as np

from .rot import axis_angle_to_mat, mm3, mv3


class Kinematics:
    def __init__(self, meta: dict):
        self.parent = list(meta["kin_parent"])
        self.joint_idx = list(meta["kin_joint_idx"])
        self.axes = np.asarray(meta["kin_axes"], np.float32)
        self.off_p = np.asarray(meta["kin_off_p"], np.float32)
        self.off_R = np.asarray(meta["kin_off_R"], np.float32)
        self.tracked_idx = list(meta["kin_tracked_idx"])
        self.num_bodies = len(self.parent)
        self.q_lo = np.asarray(meta["q_lo"], np.float32)
        self.q_hi = np.asarray(meta["q_hi"], np.float32)
        # Absent metadata deliberately retains the original floating-root hinge
        # arithmetic, including its exact operation order.
        self.joint_kind = meta.get("kin_joint_kind")
        if self.joint_kind is not None:
            self.joint_kind = tuple(self.joint_kind)
            self.joint_scale = np.asarray(meta["kin_joint_scale"], np.float32)
            self.joint_bias = np.asarray(meta["kin_joint_bias"], np.float32)
            self.anchor = np.asarray(meta["kin_anchor"], np.float32)
            if (len(self.joint_kind) != self.num_bodies or any(k not in ('fixed', 'hinge', 'slide') for k in self.joint_kind)
                    or self.joint_scale.shape != (self.num_bodies,) or self.joint_bias.shape != (self.num_bodies,)
                    or self.anchor.shape != (self.num_bodies, 3)):
                raise ValueError('invalid scalar-joint kinematics metadata')

    @classmethod
    def from_mujoco(cls, model, joint_names, *, dependent_joints=(), tracked_bodies=None):
        """Compile a fixed-base scalar tree from the owner's compiled projection.

        Native ordering is explicit. Other moving joints require declared affine
        leader relations; no joint-name or anatomical inference is performed.
        Multiple scalar joints in one body become internal tree nodes, not new
        material bodies. MuJoCo joint anchors and reference coordinates survive.
        """
        import mujoco

        names = tuple(joint_names)
        if len(set(names)) != len(names):
            raise ValueError('native coordinates must be distinct')
        ids = [model.joint(name).id for name in names]
        declared = {item['joint']: item for item in dependent_joints}
        if len(declared) != len(dependent_joints) or set(declared) & set(names):
            raise ValueError('dependent coordinates cannot duplicate native coordinates')
        all_names = {model.joint(i).name for i in range(model.njnt)}
        if set(names) | set(declared) != all_names:
            raise ValueError('every moving coordinate needs a native or dependent binding')
        def resolve(name, visiting=()):
            if name in names:
                return names.index(name), 1., 0.
            if name in visiting or name not in declared:
                raise ValueError('cyclic or unknown dependent leader')
            item = declared[name]
            index, scale, bias = resolve(item['leader'], (*visiting, name))
            multiplier, offset = float(item['multiplier']), float(item['offset'])
            if not np.isfinite([multiplier, offset]).all():
                raise ValueError('nonfinite dependent relation')
            return index, scale*multiplier, bias*multiplier+offset
        parent, joint_idx, axes = [-1], [-1], [np.zeros(3)]
        off_p, off_R, anchors = [np.zeros(3)], [np.eye(3)], [np.zeros(3)]
        kinds, scales, biases = ['fixed'], [0.], [0.]
        body_nodes = [0]*model.nbody
        for body in range(1, model.nbody):
            count = int(model.body_jntnum[body])
            joints = range(int(model.body_jntadr[body]), int(model.body_jntadr[body])+count) if count else (None,)
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, model.body_quat[body])
            par = body_nodes[int(model.body_parentid[body])]
            for ordinal, joint in enumerate(joints):
                parent.append(par)
                off_p.append(model.body_pos[body].copy() if ordinal == 0 else np.zeros(3))
                off_R.append(rotation.reshape(3, 3).copy() if ordinal == 0 else np.eye(3))
                if joint is None:
                    kind, index, scale, bias = 'fixed', -1, 0., 0.
                    axis, anchor = np.zeros(3), np.zeros(3)
                else:
                    kind = {int(mujoco.mjtJoint.mjJNT_HINGE): 'hinge', int(mujoco.mjtJoint.mjJNT_SLIDE): 'slide'}.get(int(model.jnt_type[joint]))
                    if kind is None:
                        raise ValueError('stationary scalar FK does not support free or ball joints')
                    index, scale, bias = resolve(model.joint(joint).name)
                    bias -= float(model.qpos0[model.jnt_qposadr[joint]])
                    axis, anchor = model.jnt_axis[joint].copy(), model.jnt_pos[joint].copy()
                kinds.append(kind)
                joint_idx.append(index)
                scales.append(scale)
                biases.append(bias)
                axes.append(axis)
                anchors.append(anchor)
                par = len(parent)-1
            body_nodes[body] = par
        tracked = list(range(model.nbody)) if tracked_bodies is None else [model.body(name).id for name in tracked_bodies]
        result = cls(dict(kin_parent=parent, kin_joint_idx=joint_idx, kin_axes=axes, kin_off_p=off_p,
                          kin_off_R=off_R, kin_tracked_idx=[body_nodes[i] for i in tracked],
                          q_lo=model.jnt_range[ids, 0], q_hi=model.jnt_range[ids, 1],
                          kin_joint_kind=kinds, kin_joint_scale=scales, kin_joint_bias=biases, kin_anchor=anchors))
        result.body_nodes = tuple(body_nodes)
        return result

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions (..., n, 3), rotations (..., n, 3, 3), link-origin linear and angular velocities."""
        P, Rs, V, W = [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies
        for i in range(self.num_bodies):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            Rp = Rs[par]
            pi = P[par] + mv3(Rp, self.off_p[i])
            R_off = mm3(Rp, self.off_R[i])
            if self.joint_kind is not None:
                kind = self.joint_kind[i]
                if kind == 'fixed':
                    Ri, wi = R_off, W[par]
                    vi = V[par] + jnp.cross(W[par], pi-P[par])
                else:
                    angle = q[..., j]*self.joint_scale[i]+self.joint_bias[i]
                    rate = qd[..., j:j+1]*self.joint_scale[i]
                    axis_world = mv3(R_off, self.axes[i])
                    if kind == 'slide':
                        pi = pi+axis_world*angle[..., None]
                        Ri, wi = R_off, W[par]
                        vi = V[par]+jnp.cross(W[par], pi-P[par])+axis_world*rate
                    else:
                        Ri = mm3(R_off, axis_angle_to_mat(self.axes[i], angle))
                        lever = mv3(Ri, self.anchor[i])
                        pi = pi+mv3(R_off, self.anchor[i])-lever
                        wi = W[par]+axis_world*rate
                        vi = V[par]+jnp.cross(W[par], pi-P[par])-jnp.cross(axis_world*rate, lever)
                P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
                continue
            Ri = R_off if j < 0 else mm3(R_off, axis_angle_to_mat(self.axes[i], q[..., j]))
            wi = W[par] if j < 0 else W[par] + mv3(R_off, self.axes[i]) * qd[..., j : j + 1]
            vi = V[par] + jnp.cross(W[par], pi - P[par])
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (jnp.stack([P[i] for i in idx], -2), jnp.stack([Rs[i] for i in idx], -3),
                jnp.stack([V[i] for i in idx], -2), jnp.stack([W[i] for i in idx], -2))  # fmt: skip
