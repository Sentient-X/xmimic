"""Fixed-base URR physics tasks and an evaluator independent of learner rewards.

Canonical bodies own geometry, joint units, sites and aperture laws. This module
consumes a prepared nominal projection; it never infers anatomy from joint names.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor
import copy
import os
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from sx_actions.urr import Motion, MotionBinding, MotionFrame, motion_to_dict
from sx_embodiments import Embodiment
from sx_embodiments.kinematics import Kinematics
from sx_embodiments.parts import GripperSpec
from sx_geometry import FramedTransform, Pose, mat_to_quat, orientation_error

from .artifact_cache import _atomic, _json, content_hash
from .kin import Kinematics as DifferentiableKinematics
from . import rot


@dataclass(frozen=True)
class TaskConfig:
    episode_steps: int = 100
    precision: bool = False
    accumulate: bool = False
    partition: str = "train"
    position_tolerance_m: float = .001
    orientation_tolerance_rad: float = float(np.deg2rad(.2))
    aperture_tolerance_m: float = .001
    hold_steps: int = 25
    target_fraction: float = .08
    moving: bool = False

    def __post_init__(self):
        if (self.episode_steps < 2 or self.hold_steps < 1
                or self.hold_steps > self.episode_steps or self.partition not in {"train", "evaluation"}
                or not 0 < self.target_fraction <= .5
                or min(self.position_tolerance_m, self.orientation_tolerance_rad,
                       self.aperture_tolerance_m) <= 0):
            raise ValueError("Invalid frozen benchmark task settings")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rotation(R):
    return np.asarray(R, dtype=np.float64).reshape(3, 3)


def _safe_norm(value, axis=-1):
    """Euclidean norm with its zero subgradient defined at an exact target."""
    square = jnp.sum(jnp.square(value), axis=axis)
    positive = square > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, square, 1.)), 0.)


class URRPhysicsEnv:
    """One nominal fixed-base simulator; the learner sees tasks, never target joints."""

    action_rate_weight = .001

    def __init__(self, prepared: Path, config: TaskConfig = TaskConfig()):
        self.prepared, self.config = Path(prepared).resolve(), config
        self.spec = json.loads((self.prepared / "benchmark.json").read_bytes())
        self.body = Embodiment.from_json((self.prepared / "embodiment.json").read_text())
        if str(self.body.id) != self.spec["embodiment_id"]:
            raise ValueError("Benchmark body identity mismatch")
        self.names = tuple(self.body.joint_names)
        if tuple(self.spec["joint_names"]) != self.names:
            raise ValueError("Benchmark coordinates must retain canonical native ordering")
        scene = self.prepared / "robot" / "scene.xml"
        if digest(scene) != self.spec["scene_sha256"] or hashlib.sha256(self.body.urdf_bytes).hexdigest() != self.spec["urdf_sha256"]:
            raise ValueError("Prepared scene or canonical URDF changed")
        for member in self.spec["assets"]:
            original = self.prepared / member["path"]
            path = original.resolve()
            if original.is_symlink() or not path.is_relative_to(self.prepared) or digest(path) != member["sha256"]:
                raise ValueError("Prepared canonical asset closure changed")
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        if np.any(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError("Stationary benchmark requires a real fixed-base projection")
        self.data = mujoco.MjData(self.model)
        self.joint_ids = np.asarray([self.model.joint(n).id for n in self.names])
        self.qadr = self.model.jnt_qposadr[self.joint_ids]
        self.vadr = self.model.jnt_dofadr[self.joint_ids]
        actuator_joints = self.model.actuator_trnid[:, 0]
        controls = []
        for joint in self.joint_ids:
            matches = np.flatnonzero(actuator_joints == joint)
            if len(matches) != 1:
                raise ValueError("Every controlled coordinate needs one owner-projected actuator")
            controls.append(int(matches[0]))
        self.controls = np.asarray(controls)
        self.lo = np.asarray([c.axis.bounds.lower for c in self.body.state.coordinates])
        self.hi = np.asarray([c.axis.bounds.upper for c in self.body.state.coordinates])
        self.nominal = np.asarray(self.spec["nominal_q"], dtype=float)
        self.scale = np.asarray(self.spec["action_scale"], dtype=float)
        self.control_dt = float(self.spec["control_dt"])
        self.substeps = int(self.spec["simulation_substeps"])
        if (self.nominal.shape != self.lo.shape or self.scale.shape != self.lo.shape
                or not np.isfinite(self.nominal).all() or not np.isfinite(self.scale).all()
                or np.any(self.scale <= 0) or np.any(self.nominal < self.lo)
                or np.any(self.nominal > self.hi) or self.substeps < 1
                or not np.isclose(self.control_dt, self.model.opt.timestep * self.substeps)):
            raise ValueError("Invalid native-unit control projection")
        self.roles = dict(sorted(self.spec["roles"].items()))
        self.binding_sites = MotionBinding(self.body, self.roles)
        self.kinematics = Kinematics.of(self.body.urdf_bytes)
        (self.root_link,) = self.kinematics.links - self.kinematics.parent_joint.keys()
        self.site_frames = []
        for site_name in self.roles.values():
            site = self.body.site(site_name)
            from sx_embodiments.kinematics import from_xyz_rpy
            self.site_frames.append((self.model.body(site.link).id,
                                     np.asarray(from_xyz_rpy(site.pose.xyz, site.pose.rpy)).reshape(4, 4)))
        components = {str(c.component_id): c for c in self.body.components}
        self.apertures = []
        for role, aperture in sorted(self.spec["apertures"].items()):
            component = components[aperture["component_id"]]
            if not isinstance(component.part, GripperSpec):
                raise ValueError("Aperture role must identify an owner gripper")
            index = self.names.index(aperture["drive_joint"])
            # Refuse missing owner relations before allocating training work.
            component.part.aperture_from_drive(float(self.nominal[index]))
            self.apertures.append((role, component.part, index))
        self.action_dim = len(self.names)
        self.goal_dim = 9 * len(self.roles) + len(self.apertures)
        self.error_dim = 6 * len(self.roles) + len(self.apertures)
        self.goal_slice = slice(4 * self.action_dim, 4 * self.action_dim + self.goal_dim)
        self.error_slice = slice(self.goal_slice.stop, self.goal_slice.stop + self.error_dim)
        self.coarse = np.asarray([v for _ in self.roles for v in [.01] * 3 + [.35] * 3]
                                 + [.01] * len(self.apertures))
        self.tolerances = np.asarray([v for _ in self.roles for v in
                                     [config.position_tolerance_m] * 3 + [config.orientation_tolerance_rad] * 3]
                                    + [config.aperture_tolerance_m] * len(self.apertures))
        self.observation_dim = self.error_slice.stop + (2 * self.error_dim if config.precision else 0) + 3
        self.physical_state_dim = 2*self.action_dim
        self.projection_kinematics = DifferentiableKinematics.from_mujoco(
            self.model, self.names, dependent_joints=self.spec.get('dependent_joints', ()),
            tracked_bodies=[self.model.body(body_id).name for body_id, _ in self.site_frames])
        self.aperture_curves = []
        for _, part, index in self.apertures:
            if part.gap_curve is not None:
                xs = [k.x for k in part.gap_curve.knots]
                ys = [k.y for k in part.gap_curve.knots]
            elif part.travel_m is not None and part.dof == 1:
                bounds = part.layout.axes[0].bounds
                xs, ys = [bounds.lower, bounds.upper], list(part.travel_m)
            else:
                raise ValueError('Differentiable aperture needs a declared curve or affine travel law')
            self.aperture_curves.append((index, np.asarray(xs), np.asarray(ys)))
        self.binding = {"schema": "xmimic.urr-physics-task/v1", "body_id": str(self.body.id),
                        "scene_sha256": digest(scene), "preparation_sha256": digest(self.prepared / "benchmark.json"),
                        "body_document_sha256": digest(self.prepared / "embodiment.json"),
                        "environment_code_sha256": digest(Path(__file__)), "task": asdict(config),
                        "owner_code": {cls.__module__: digest(Path(sys.modules[cls.__module__].__file__))
                                       for cls in (Embodiment, Kinematics, GripperSpec, MotionBinding, DifferentiableKinematics)},
                        "projection": "physical-q-qd/canonical-observation-v1", "rotation_code_sha256": digest(Path(rot.__file__)),
                        "observation": "native_q_normalized,qd_scaled,command_error_scaled,previous_action,URR_goal_3p6R_aperture,raw_error,optional_coarse_fine,progress_dwell",
                        "action": "bounded native-unit joint-reference increments" if config.accumulate else "full-authority nominal-relative references with shared native-unit rate governor",
                        "observation_dim": self.observation_dim, "action_dim": self.action_dim,
                        "scope": "nominal simulation; FK-generated task goals; no physical qualification"}
        self.command = self.nominal.copy()
        self.previous_action = np.zeros(self.action_dim)
        self.steps = 0
        self.goal = None
        self.rng = np.random.default_rng(0)
        self.best = 0.
        self.progress = 0.
        self.dwell = 0

    def _set_positions(self, data, q):
        data.qpos[self.qadr] = q
        positions = dict(zip(self.names, q, strict=True))
        for dependent in self.spec.get("dependent_joints", []):
            value = dependent["multiplier"] * positions[dependent["leader"]] + dependent["offset"]
            data.qpos[self.model.jnt_qposadr[self.model.joint(dependent["joint"]).id]] = value
            positions[dependent["joint"]] = value
        mujoco.mj_forward(self.model, data)

    def observe(self) -> MotionFrame:
        poses = {}
        for role, (body_id, offset) in zip(self.roles, self.site_frames, strict=True):
            R = _rotation(self.data.xmat[body_id])
            poses[role] = Pose(self.data.xpos[body_id] + R @ offset[:3, 3], mat_to_quat(R @ offset[:3, :3]))
        widths = {role: part.aperture_from_drive(float(self.data.qpos[self.qadr[index]]))
                  for role, part, index in self.apertures}
        return MotionFrame(round(self.steps * self.control_dt * 1e9), "world", poses, widths)

    def _goal_for(self, q) -> MotionFrame:
        positions = dict(zip(self.names, q, strict=True))
        root_id = self.model.body(self.root_link).id
        reference = FramedTransform("world", self.root_link,
                                    Pose(self.data.xpos[root_id].copy(), mat_to_quat(_rotation(self.data.xmat[root_id]))))
        observed = self.binding_sites.observe(time_ns=0, positions=positions, reference=reference)
        return MotionFrame(0, "world", observed.poses,
                           {role: part.aperture_from_drive(float(q[index])) for role, part, index in self.apertures})

    def set_goal(self, goal: MotionFrame):
        if (goal.frame != "world" or set(goal.poses) != set(self.roles)
                or set(goal.apertures) != {a[0] for a in self.apertures}
                or any(not isinstance(p, Pose) for p in goal.poses.values())):
            raise ValueError("Task needs exactly its declared world-frame pose/aperture observations")
        self.goal = goal
        self.best = float(self._ratio(self.errors()))
        self.progress, self.dwell = 0., 0

    def reset(self, seed: int = 0) -> np.ndarray:
        if type(seed) is not int or seed < 0:
            raise ValueError("Task seed must be a non-negative integer")
        # Disjoint seed namespaces; the same seed cannot mint train and test tasks.
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, int(self.config.partition == "evaluation")]))
        mujoco.mj_resetData(self.model, self.data)
        width = self.hi - self.lo
        self.start_q = np.clip(self.nominal + self.rng.uniform(-.02, .02, self.action_dim) * width, self.lo, self.hi)
        self.target_q = np.clip(self.nominal + self.rng.uniform(-1, 1, self.action_dim) * width * self.config.target_fraction,
                                self.lo + .01 * width, self.hi - .01 * width)
        for _, _, index in self.apertures:
            self.target_q[index] = self.rng.uniform(self.lo[index] + .1 * width[index], self.hi[index] - .1 * width[index])
        self.command = self.start_q.copy()
        self.previous_action = np.zeros(self.action_dim)
        self.steps = 0
        self._set_positions(self.data, self.start_q)
        self.data.ctrl[self.controls] = self.command
        self.set_goal(self._goal_for(self.start_q if self.config.moving else self.target_q))
        return self.observation()

    def errors(self) -> np.ndarray:
        actual = self.observe()
        errors = []
        for role in self.roles:
            desired, achieved = self.goal.poses[role], actual.poses[role]
            errors.extend(desired.position - achieved.position)
            errors.extend(orientation_error(achieved.quaternion, desired.quaternion))
        errors.extend(self.goal.apertures[role] - actual.apertures[role] for role, _, _ in self.apertures)
        return np.asarray(errors)

    def observation(self) -> np.ndarray:
        q, qd = self.data.qpos[self.qadr], self.data.qvel[self.vadr]
        goal = []
        from sx_geometry import quat_to_mat
        for role in self.roles:
            pose = self.goal.poses[role]
            goal.extend(pose.position)
            goal.extend(np.asarray(quat_to_mat(pose.quaternion))[:, :2].T.ravel())
        goal.extend(self.goal.apertures[role] for role, _, _ in self.apertures)
        error = self.errors()
        values = [(q - (self.hi + self.lo) / 2) / ((self.hi - self.lo) / 2),
                  qd * self.control_dt / self.scale, (self.command - q) / self.scale,
                  self.previous_action, np.asarray(goal), error]
        if self.config.precision:
            values.extend([np.clip(error / self.coarse, -1, 1), np.clip(error / self.tolerances, -1, 1)])
        values.append(np.asarray([self.best / 100., self.progress, self.dwell / self.config.hold_steps]))
        result = np.concatenate(values).astype(np.float32)
        if result.shape != (self.observation_dim,) or not np.isfinite(result).all():
            raise ValueError("Nonfinite or malformed benchmark observation")
        return result

    def _ratio(self, error):
        error = jnp.asarray(error)
        pose = error[..., :6 * len(self.roles)].reshape(error.shape[:-1] + (len(self.roles), 6))
        ratios = [jnp.max(_safe_norm(pose[..., :3], axis=-1) / self.config.position_tolerance_m, axis=-1),
                  jnp.max(_safe_norm(pose[..., 3:], axis=-1) / self.config.orientation_tolerance_rad, axis=-1)]
        if self.apertures:
            ratios.append(jnp.max(jnp.abs(error[..., 6 * len(self.roles):]) / self.config.aperture_tolerance_m, axis=-1))
        return jnp.max(jnp.stack(ratios, axis=-1), axis=-1)

    def reward_from_observation(self, observation, previous_action=None):
        obs = jnp.asarray(observation)
        error = obs[..., self.error_slice]
        coarse = jnp.asarray(self.coarse)
        reward = jnp.mean(jnp.exp(-jnp.square(error / coarse)), axis=-1)
        if self.config.precision:
            ratio = self._ratio(error)
            reward = reward + 1 - jnp.tanh(ratio) + obs[..., -2] + .1 * obs[..., -1]
        if previous_action is not None:
            action = obs[..., 3 * self.action_dim:4 * self.action_dim]
            reward = reward - self.action_rate_weight * jnp.mean(jnp.square(action - jnp.asarray(previous_action)), axis=-1)
        return reward

    def _command_reference(self, command, action, error, xp, *, ratio=None):
        """One command transition shared by physics and differentiable imagination."""
        if self.config.precision:
            if ratio is None:
                ratio = np.asarray(self._ratio(error), dtype=float) if xp is np else self._ratio(error)
            gain = .1+.9*xp.clip((ratio-1)/19., 0., 1.)
        else:
            gain = xp.asarray(1.)
        if self.config.accumulate:
            desired = command+self.scale*gain[..., None]*action
        else:
            authority = np.maximum(self.hi-self.nominal, self.nominal-self.lo)
            desired = self.nominal+authority*action
        desired = xp.clip(desired, command-self.scale, command+self.scale)
        return xp.clip(desired, self.lo, self.hi)

    def project_observation(self, previous, action, predicted):
        """Learn q/qd only; derive every other stationary task channel from facts.

        `predicted` retains the full network width for checkpoint shape clarity,
        but its command, goal, errors and history channels have no authority.
        Moving goals require explicit trajectory state and are refused here.
        """
        if self.config.moving:
            raise ValueError('Structured world-model training requires stationary goals; moving trajectory state is not represented')
        previous, predicted = jnp.asarray(previous), jnp.asarray(predicted)
        action = jnp.clip(jnp.asarray(action), -1., 1.)
        J = self.action_dim
        midpoint, half = (self.hi+self.lo)/2, (self.hi-self.lo)/2
        old_q = previous[..., :J]*half+midpoint
        command = old_q+previous[..., 2*J:3*J]*self.scale
        command = self._command_reference(command, action, previous[..., self.error_slice], jnp)
        q = predicted[..., :J]*half+midpoint
        qd = predicted[..., J:2*J]*self.scale/self.control_dt
        shape = q.shape[:-1]
        zero = jnp.zeros(shape+(3,), dtype=q.dtype)
        identity = jnp.broadcast_to(jnp.eye(3, dtype=q.dtype), shape+(3, 3))
        positions, rotations, _, _ = self.projection_kinematics.forward(zero, identity, zero, zero, q, qd)
        return self._observation_channels(previous, action, predicted[..., :2*J], q, command, positions, rotations)

    def _observation_channels(self, previous, action, physical, q, command, positions, rotations):
        """Shared known task channels for actual MuJoCo frames and imagined FK."""
        goal = previous[..., self.goal_slice]
        errors = []
        for i, (_, offset) in enumerate(self.site_frames):
            position = positions[..., i, :]+rot.mv3(rotations[..., i, :, :], offset[:3, 3])
            rotation = rot.mm3(rotations[..., i, :, :], offset[:3, :3])
            first, second = goal[..., 9*i+3:9*i+6], goal[..., 9*i+6:9*i+9]
            desired = jnp.stack((first, second, jnp.cross(first, second)), axis=-1)
            errors.extend((goal[..., 9*i:9*i+3]-position,
                           rot.so3_log(rot.mm3(desired, rot.tr3(rotation)), safe_zero=True)))
        for i, (index, xs, ys) in enumerate(self.aperture_curves):
            aperture = jnp.interp(q[..., index], jnp.asarray(xs), jnp.asarray(ys))
            errors.append((goal[..., 9*len(self.roles)+i]-aperture)[..., None])
        error = jnp.concatenate(errors, axis=-1)
        values = [physical, (command-q)/self.scale, action, goal, error]
        if self.config.precision:
            values.extend((jnp.clip(error/self.coarse, -1., 1.), jnp.clip(error/self.tolerances, -1., 1.)))
        ratio = self._ratio(error)
        best = previous[..., -3]*100.
        progress = jnp.maximum(0., best-ratio)/100.
        dwell = jnp.where(ratio <= 1., previous[..., -1]+1./self.config.hold_steps, 0.)
        values.append(jnp.stack((jnp.minimum(best, ratio)/100., progress, dwell), axis=-1))
        return jnp.concatenate(values, axis=-1)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError("Action must be finite and match native coordinates")
        action = np.clip(action, -1, 1)
        self.command = self._command_reference(self.command, action, self.errors(), np)
        previous = self.previous_action.copy()
        self.previous_action = action.copy()
        self.data.ctrl[self.controls] = self.command
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        # mj_step integrates qpos after its last position stage. Refresh FK so
        # achieved poses, forces and timestamps describe that returned state.
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        if self.config.moving:
            moving_steps = max(1, self.config.episode_steps-self.config.hold_steps)
            if self.steps <= moving_steps:
                fraction = self.steps/moving_steps
                # A different goal has different progress/holding history.
                self.set_goal(self._goal_for(self.start_q+fraction*(self.target_q-self.start_q)))
        error = self.errors()
        ratio = float(self._ratio(error))
        self.progress = max(0., self.best - ratio) / 100.
        self.best = min(self.best, ratio)
        self.dwell = self.dwell + 1 if self._within(error) else 0
        obs = self.observation()
        done = self.steps >= self.config.episode_steps
        return obs, float(self.reward_from_observation(obs, previous)), done, self.measurement(error)

    def _within(self, error) -> bool:
        position = error[:6 * len(self.roles)].reshape(-1, 6)
        return bool(np.all(np.linalg.norm(position[:, :3], axis=1) <= self.config.position_tolerance_m)
                    and np.all(np.linalg.norm(position[:, 3:], axis=1) <= self.config.orientation_tolerance_rad)
                    and np.all(np.abs(error[6 * len(self.roles):]) <= self.config.aperture_tolerance_m))

    def measurement(self, error=None) -> dict:
        error = self.errors() if error is None else error
        pose = error[:6 * len(self.roles)].reshape(-1, 6)
        q = self.data.qpos[self.qadr]
        return {"position_mm": (np.linalg.norm(pose[:, :3], axis=1) * 1000).tolist(),
                "orientation_deg": np.rad2deg(np.linalg.norm(pose[:, 3:], axis=1)).tolist(),
                "aperture_mm": (np.abs(error[6 * len(self.roles):]) * 1000).tolist(),
                "within_tolerance": self._within(error), "dwell_steps": self.dwell,
                "joint_limit_violation": bool(np.any(q < self.lo - 1e-6) or np.any(q > self.hi + 1e-6)),
                "contacts": int(self.data.ncon), "physics_warning_count": int(sum(w.number for w in self.data.warning)),
                "native_velocity": self.data.qvel[self.vadr].tolist(),
                "native_actuator_force": self.data.actuator_force[self.controls].tolist(),
                "time_s": float(self.data.time)}

    def state(self):
        state_size = mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION)
        physics = np.zeros(state_size)
        mujoco.mj_getState(self.model, self.data, physics, mujoco.mjtState.mjSTATE_INTEGRATION)
        rng_bytes = json.dumps(self.rng.bit_generator.state).encode()
        if len(rng_bytes) > 1024:
            raise ValueError("RNG state exceeds fixed checkpoint encoding")
        rng_buffer = np.zeros(1024, dtype=np.uint8)
        rng_buffer[:len(rng_bytes)] = np.frombuffer(rng_bytes, dtype=np.uint8)
        return {"physics": physics, "command": self.command.copy(), "previous_action": self.previous_action.copy(),
                "steps": np.asarray(self.steps), "best": np.asarray(self.best), "progress": np.asarray(self.progress),
                "dwell": np.asarray(self.dwell), "start_q": self.start_q.copy(), "target_q": self.target_q.copy(),
                "rng_json": rng_buffer, "rng_length": np.asarray(len(rng_bytes)),
                "goal_position": np.asarray([self.goal.poses[role].position for role in self.roles]),
                "goal_quaternion": np.asarray([self.goal.poses[role].quaternion for role in self.roles]),
                "goal_aperture": np.asarray([self.goal.apertures[role] for role, _, _ in self.apertures]),
                "goal_time_ns": np.asarray(self.goal.time_ns, dtype=np.int64)}

    def restore(self, state):
        mujoco.mj_setState(self.model, self.data, state["physics"], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model, self.data)
        self.command = np.asarray(state["command"]).copy()
        self.previous_action = np.asarray(state["previous_action"]).copy()
        self.steps, self.dwell = int(state["steps"]), int(state["dwell"])
        self.best, self.progress = float(state["best"]), float(state["progress"])
        self.start_q, self.target_q = np.asarray(state["start_q"]).copy(), np.asarray(state["target_q"]).copy()
        self.rng.bit_generator.state = json.loads(np.asarray(state["rng_json"], dtype=np.uint8)[:int(state["rng_length"])].tobytes())
        self.goal = MotionFrame(int(state["goal_time_ns"]), "world",
            {role: Pose(state["goal_position"][i], state["goal_quaternion"][i]) for i, role in enumerate(self.roles)},
            {role: float(state["goal_aperture"][i]) for i, (role, _, _) in enumerate(self.apertures)})

    def vectorize(self, num_envs):
        return URRPhysicsBatch(self, num_envs)

    def ik_action(self, damping: float = .01, feedback_gain: float = .1):
        """Bounded numerical control baseline, using actual MuJoCo Jacobians."""
        jacobians, errors = [], []
        error = self.errors()
        for i, (body_id, offset) in enumerate(self.site_frames):
            R = _rotation(self.data.xmat[body_id])
            point = self.data.xpos[body_id] + R @ offset[:3, 3]
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jp, jr, point, body_id)
            jacobians.extend([jp[:, self.vadr] / self.coarse[6 * i], jr[:, self.vadr] / self.coarse[6 * i + 3]])
            errors.extend([error[6 * i:6 * i + 3] / self.coarse[6 * i], error[6 * i + 3:6 * i + 6] / self.coarse[6 * i + 3]])
        for i, (_, part, index) in enumerate(self.apertures):
            q = float(self.data.qpos[self.qadr[index]])
            eps = max(1e-8, (self.hi[index] - self.lo[index]) * 1e-5)
            derivative = (part.aperture_from_drive(q + eps) - part.aperture_from_drive(q - eps)) / (2 * eps)
            row = np.zeros((1, self.action_dim))
            row[0, index] = derivative / .01
            jacobians.append(row)
            errors.append(np.asarray([error[6 * len(self.roles) + i] / .01]))
        J, e = np.concatenate(jacobians), np.concatenate(errors)
        # Optimize the increment inside native limits rather than clip an unconstrained solve.
        from scipy.optimize import lsq_linear
        lower = np.maximum(self.lo - self.command, -self.scale)
        upper = np.minimum(self.hi - self.command, self.scale)
        solve = lsq_linear(np.vstack([J, damping * np.eye(self.action_dim)]),
                           np.concatenate([e, np.zeros(self.action_dim)]), bounds=(lower - 1e-12, upper + 1e-12))
        desired = np.clip(self.command + feedback_gain * solve.x, self.lo, self.hi)
        reference = self.command if self.config.accumulate else self.nominal
        authority = self.scale if self.config.accumulate else np.maximum(self.hi - self.nominal, self.nominal - self.lo)
        return np.clip((desired - reference) / authority, -1, 1)


class URRPhysicsBatch:
    """Independent MuJoCo integrations sharing only immutable model/robot facts.

    The bounded pool advances existing MjData in place, including warmstart and
    solver state. Known observation channels are compiled once for the batch;
    measured frames come from MuJoCo, never from a learned dynamics proposal.
    """

    def __init__(self, template, num_envs):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError('num_envs must be a positive integer')
        if template.config.moving:
            raise ValueError('Batched training requires stationary goals')
        self.template, self.num_envs = template, num_envs
        self.action_dim, self.observation_dim = template.action_dim, template.observation_dim
        self.binding = template.binding
        self.envs = []
        for _ in range(num_envs):
            env = copy.copy(template)
            env.data = mujoco.MjData(template.model)
            env.rng = np.random.default_rng(0)
            self.envs.append(env)
        self.workers = min(num_envs, 8, os.cpu_count() or 1)
        self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='urr-physics')
        self.groups = [self.envs[i::self.workers] for i in range(self.workers)]
        self._ratio = jax.jit(template._ratio)
        self._channels = jax.jit(template._observation_channels)
        self._reward = jax.jit(template.reward_from_observation)
        self.reward_from_observation = template.reward_from_observation
        self.project_observation = template.project_observation
        self.observations = np.empty((num_envs, self.observation_dim), np.float32)

    def reset(self, seeds):
        seeds = np.asarray(seeds)
        if seeds.shape != (self.num_envs,):
            raise ValueError('One explicit seed is required per environment')
        self.observations = np.stack([env.reset(int(seed)) for env, seed in zip(self.envs, seeds, strict=True)])
        return self.observations.copy()

    def reset_at(self, indices, seeds):
        for index, seed in zip(indices, seeds, strict=True):
            self.observations[index] = self.envs[index].reset(int(seed))
        return self.observations.copy()

    @staticmethod
    def _advance(group):
        for env in group:
            mujoco.mj_step(env.model, env.data, nstep=env.substeps)
            mujoco.mj_forward(env.model, env.data)

    def step(self, actions):
        t = self.template
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.num_envs, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError('Actions must be a finite batch of native coordinates')
        actions = np.clip(actions, -1., 1.)
        previous = self.observations.copy()
        command = np.stack([env.command for env in self.envs])
        ratio = np.asarray(self._ratio(previous[:, t.error_slice]), dtype=float) if t.config.precision else None
        command = t._command_reference(command, actions, previous[:, t.error_slice], np, ratio=ratio)
        for env, reference, action in zip(self.envs, command, actions, strict=True):
            env.command = reference.copy()
            env.previous_action = action.copy()
            env.data.ctrl[env.controls] = reference
        # Consume every future so exceptions cannot masquerade as completed work.
        list(self.pool.map(self._advance, self.groups))
        q = np.stack([env.data.qpos[t.qadr] for env in self.envs])
        qd = np.stack([env.data.qvel[t.vadr] for env in self.envs])
        ids = [body_id for body_id, _ in t.site_frames]
        positions = np.stack([env.data.xpos[ids] for env in self.envs])
        rotations = np.stack([env.data.xmat[ids].reshape(-1, 3, 3) for env in self.envs])
        physical = np.concatenate(((q-(t.hi+t.lo)/2)/((t.hi-t.lo)/2), qd*t.control_dt/t.scale), axis=-1)
        self.observations = np.asarray(self._channels(previous, actions, physical, q, command, positions, rotations)).copy()
        rewards = np.asarray(self._reward(self.observations, previous[:, 3*t.action_dim:4*t.action_dim]))
        for env, obs in zip(self.envs, self.observations, strict=True):
            env.steps += 1
            env.best, env.progress = float(obs[-3])*100., float(obs[-2])
            env.dwell = round(float(obs[-1])*env.config.hold_steps)
        done = np.asarray([env.steps >= env.config.episode_steps for env in self.envs])
        warnings = np.asarray([sum(w.number for w in env.data.warning) for env in self.envs])
        return self.observations.copy(), rewards, done, {'physics_warning_count': warnings}

    def state(self):
        states = [env.state() for env in self.envs]
        return {**{key: np.stack([state[key] for state in states]) for key in states[0]},
                'batch_observations': self.observations.copy()}

    def restore(self, state):
        for index, env in enumerate(self.envs):
            env.restore({key: value[index] for key, value in state.items() if key != 'batch_observations'})
        self.observations = np.asarray(state['batch_observations']).copy()

    def close(self):
        self.pool.shutdown(wait=True)


def evaluate(env_factory: Callable[[], URRPhysicsEnv], policy, seeds: list[int], output: Path, *, policy_identity: str | None = None):
    """Retain every achieved sample and source target; no learner reward determines success."""
    output = Path(output)
    if "raw" in output.resolve().parts or output.is_symlink():
        raise ValueError("Evaluation requires a regular derived output directory")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Evaluation needs distinct non-negative task seeds")
    if policy is not None and not policy_identity:
        raise ValueError("Learned evaluation requires an immutable policy identity")
    output.mkdir(parents=True, exist_ok=True)
    env = env_factory()
    if env.config.partition != "evaluation":
        raise ValueError("Evaluation must use its disjoint task partition")
    request = {"environment": env.binding, "seeds": seeds, "schema": "xmimic.urr-evaluation/v1",
               "policy_identity": policy_identity or "bounded-numerical-IK:" + digest(Path(__file__))}
    request_file = output / "request.json"
    if request_file.exists() and json.loads(request_file.read_bytes()) != request:
        raise ValueError("Evaluation directory belongs to another immutable request")
    _atomic(request_file, _json(request))
    request_hash = content_hash(request)
    completed = []
    for seed in seeds:
        destination = output / f"task-{seed}.json"
        if destination.exists():
            retained = json.loads(destination.read_bytes())
            if (retained.get("request_sha256") != request_hash or retained.get("seed") != seed
                    or len(retained.get("measurements", [])) != env.config.episode_steps):
                raise ValueError("Retained task belongs to another evaluation or is incomplete")
            completed.append(retained)
            continue
        obs = env.reset(seed)
        records, observations, goals, coordinates = [], [], [], []
        for _ in range(env.config.episode_steps):
            action = env.ik_action() if policy is None else policy(obs)
            obs, _, _, metrics = env.step(action)
            records.append(metrics)
            observations.append(env.observe())
            goals.append(MotionFrame(observations[-1].time_ns, env.goal.frame, env.goal.poses, env.goal.apertures))
            coordinates.append(env.data.qpos[env.qadr].copy())
        result = {"request_sha256": request_hash, "seed": seed, "measurements": records, "achieved": motion_to_dict(Motion(tuple(observations))),
                  "targets": motion_to_dict(Motion(tuple(goals))), "native_q": np.asarray(coordinates).tolist(),
                  "held": all(r["within_tolerance"] for r in records[-env.config.hold_steps:]),
                  "constraint_failed": any(r["joint_limit_violation"] or r["physics_warning_count"] for r in records)}
        _atomic(destination, _json(result))
        completed.append(result)
    def summarize(records):
        result = {}
        for key in ("position_mm", "orientation_deg", "aperture_mm"):
            values = np.asarray([r[key] for r in records])
            result[key] = {"median": np.median(values, axis=0).tolist(), "p95": np.quantile(values, .95, axis=0).tolist(),
                           "max": np.max(values, axis=0).tolist()}
        return result
    all_records = [r for task in completed for r in task["measurements"]]
    terminal_records = [r for task in completed for r in task["measurements"][-env.config.hold_steps:]]
    summaries = summarize(all_records)
    result = {"request_sha256": content_hash(request), "tasks": len(completed),
              "held_tasks": sum(t["held"] for t in completed), "constraint_failed_tasks": sum(t["constraint_failed"] for t in completed),
              "errors_all_steps": summaries, "errors_terminal_window": summarize(terminal_records),
              "complete": True, "training_admitted": False,
              "scope": "nominal simulator evaluation; synthetic reachable URR targets; no physical accuracy claim"}
    _atomic(output / "summary.json", _json(result))
    return result
