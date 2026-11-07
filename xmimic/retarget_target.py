"""Canonical CPU retarget targets; geometric feasibility is not training admission.

Root trajectories are explicit world-from-URDF-root poses. This solver does not
infer a human/robot root correspondence, balance, collisions, drives or dynamics.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
import math
from types import MappingProxyType
import xml.etree.ElementTree as ET

import numpy as np
from sx_contracts.identity import content_digest
from sx_embodiments import Embodiment
from sx_embodiments.assets import AssetFormat
from sx_embodiments.kinematics import Kinematics, JointType, from_xyz_rpy
from sx_embodiments.layout import Bounds, CoordinateUnit, UndocumentedDrive
from sx_geometry import chain_from_urdf, mat_to_quat, orientation_error, quat_to_mat
from sx_geometry.urdf import KinematicChain


class RetargetError(ValueError):
    """An unsupported, ambiguous or inconsistent target cannot produce a cache."""


class TargetKind(StrEnum):
    LINK = "link"
    SITE = "site"


class Disposition(StrEnum):
    GEOMETRICALLY_FEASIBLE = "geometrically_feasible"
    RESIDUAL_EXCEEDED = "residual_exceeded"


def _require(condition, message):
    if not condition:
        raise RetargetError(message)


def _readonly(value):
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _identity(value):
    return str(content_digest(value))


@dataclass(frozen=True)
class Landmark:
    source: str
    target: str
    kind: TargetKind = TargetKind.SITE
    orientation: bool = False
    weight: float = 1.0

    def __post_init__(self):
        _require(isinstance(self.source, str) and self.source.strip()
                 and isinstance(self.target, str) and self.target.strip(), "Landmarks need explicit source and target names")
        _require(isinstance(self.kind, TargetKind), "Unknown target kind")
        _require(type(self.orientation) is bool, "Orientation selection must be explicit boolean")
        _require(math.isfinite(self.weight) and self.weight > 0, "Landmark weight must be positive")


@dataclass(frozen=True)
class TargetProfile:
    landmarks: tuple[Landmark, ...]
    holds: tuple[tuple[str, float], ...] = ()
    nominal_passive: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        # Tuples ensure caller-owned lists/maps cannot change a prepared binding.
        object.__setattr__(self, "landmarks", tuple(self.landmarks))
        object.__setattr__(self, "holds", tuple((n, float(v)) for n, v in self.holds))
        object.__setattr__(self, "nominal_passive", tuple((n, float(v)) for n, v in self.nominal_passive))
        _require(self.landmarks and all(isinstance(x, Landmark) for x in self.landmarks), "Explicit landmarks required")
        _require(len({x.source for x in self.landmarks}) == len(self.landmarks), "Repeated source landmark")
        _require(len({(x.kind, x.target) for x in self.landmarks}) == len(self.landmarks), "Repeated target landmark")
        for values in (self.holds, self.nominal_passive):
            _require(len(dict(values)) == len(values), "Repeated held/passive coordinate")
            _require(all(isinstance(n, str) and n and math.isfinite(v) for n, v in values), "Invalid held/passive position")


@dataclass(frozen=True)
class TargetBinding:
    embodiment_id: str
    urdf_sha256: str
    joint_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    root_link: str
    profile: TargetProfile

    @property
    def sha256(self):
        return _identity(asdict(self))


@dataclass(frozen=True)
class PreparedTarget:
    binding: TargetBinding
    kinematics: Kinematics
    # Exact owner site offsets, or identity for an explicitly named link.
    frames: tuple[tuple[str, tuple[float, ...]], ...]
    free_indices: tuple[int, ...]
    chains: tuple[KinematicChain, ...]
    chain_coordinates: tuple[tuple[tuple[int, float, float], ...], ...]

    def poses(self, q, root_p, root_quat):
        root_R = quat_to_mat(np.asarray(root_quat, dtype=np.float64))
        out = []
        for chain, coordinates, (_, offset) in zip(self.chains, self.chain_coordinates, self.frames, strict=True):
            values = np.asarray([bias + (q[index]*scale if index >= 0 else 0.) for index, scale, bias in coordinates])
            p, quat = chain.forward(values)
            R = quat_to_mat(quat)
            site = np.asarray(offset).reshape(4, 4)
            out.append((np.asarray(root_p) + root_R @ (p + R @ site[:3, 3]), root_R @ R @ site[:3, :3]))
        return tuple(out)

    def jacobians(self, q, root_quat):
        """World spatial site Jacobians, coupled in exact canonical native order."""
        root_R = quat_to_mat(np.asarray(root_quat, dtype=np.float64))
        out = []
        for chain, coordinates, (_, offset) in zip(self.chains, self.chain_coordinates, self.frames, strict=True):
            values = np.asarray([bias + (q[index]*scale if index >= 0 else 0.) for index, scale, bias in coordinates])
            jac = chain.jacobian(values)
            _, quat = chain.forward(values)
            lever = quat_to_mat(quat) @ np.asarray(offset).reshape(4, 4)[:3, 3]
            jac[:3] += np.cross(jac[3:].T, lever).T
            full = np.zeros((6, len(self.binding.joint_names)))
            for column, (index, scale, _) in enumerate(coordinates):
                if index >= 0:
                    full[:3, index] += root_R @ jac[:3, column] * scale
                    full[3:, index] += root_R @ jac[3:, column] * scale
            out.append(full)
        return tuple(out)


def prepare_target(body: Embodiment, profile: TargetProfile, *, expected_embodiment_id: str) -> PreparedTarget:
    _require(str(body.id) == expected_embodiment_id, "Canonical embodiment identity mismatch")
    urdf = body.urdf_bytes
    sha = hashlib.sha256(urdf).hexdigest()
    _require(sha == str(body.urdf.asset.content.sha256), "Canonical URDF digest mismatch")
    kin = Kinematics.of(urdf)
    # Freeze the owner's parsed maps too; no second kinematics implementation.
    kin = Kinematics(kin.links, MappingProxyType(dict(kin.joints)), MappingProxyType(dict(kin.parent_joint)))
    roots = kin.links - kin.parent_joint.keys()
    _require(len(roots) == 1, "Target must have one connected URDF root")
    root = next(iter(roots))
    names = tuple(body.joint_names)
    _require(names and len(set(names)) == len(names), "Native names must be nonempty and unique")
    native = set(names)
    _require(native <= set(kin.independent), "Native state must name independent URDF coordinates")
    _require(all(j.type in (JointType.FIXED, JointType.REVOLUTE, JointType.CONTINUOUS)
                 for j in kin.joints.values()), "Only scalar angular target coordinates are supported")
    extra = set(kin.independent) - native
    passive = dict(profile.nominal_passive)
    _require(set(passive) == extra, f"Explicit nominal passive positions required for exactly {sorted(extra)}")
    tree = ET.fromstring(urdf)
    joints = {j.attrib["name"]: j for j in tree.findall("joint")}
    lower, upper = [], []
    for coordinate, name in zip(body.state.coordinates, names, strict=True):
        axis = coordinate.axis
        _require(axis.unit is CoordinateUnit.RADIAN and isinstance(axis.bounds, Bounds),
                 f"{name}: bounded angular native coordinate required")
        lo, hi = axis.bounds.lower, axis.bounds.upper
        _require(math.isfinite(lo) and math.isfinite(hi) and lo < hi, f"{name}: invalid bounds")
        limit = joints[name].find("limit")
        _require(limit is not None and "lower" in limit.attrib and "upper" in limit.attrib,
                 f"{name}: URDF bounds missing")
        _require(np.allclose([float(limit.attrib['lower']), float(limit.attrib['upper'])], [lo, hi], rtol=0, atol=1e-9),
                 f"{name}: native and URDF bounds disagree")
        lower.append(lo); upper.append(hi)
    for name, value in passive.items():
        limit = joints[name].find("limit")
        _require(limit is not None and "lower" in limit.attrib and "upper" in limit.attrib,
                 f"{name}: nominal passive bounds missing")
        _require(float(limit.attrib["lower"]) <= value <= float(limit.attrib["upper"]), f"{name}: passive position outside bounds")
    holds = dict(profile.holds)
    _require(set(holds) <= native, "Holds name a non-native coordinate")
    for name, value in holds.items():
        i = names.index(name)
        _require(lower[i] <= value <= upper[i], f"{name}: hold outside native bounds")
    frames, ancestors = [], set()
    for mapping in profile.landmarks:
        if mapping.kind is TargetKind.SITE:
            site = body.site(mapping.target)
            link, offset = site.link, from_xyz_rpy(site.pose.xyz, site.pose.rpy)
        else:
            link, offset = mapping.target, tuple(np.eye(4).ravel())
        _require(link in kin.links, f"Unknown target link {link}")
        _, path = kin.path(root, link)
        for joint in path:
            if joint.type is JointType.FIXED:
                continue
            while joint.mimic is not None:
                joint = kin.joints[joint.mimic.leader]
            ancestors.add(joint.name)
        frames.append((link, tuple(offset)))
    unmatched = native - ancestors
    _require(unmatched <= set(holds), f"Unmatched coordinates need explicit holds: {sorted(unmatched-set(holds))}")
    binding = TargetBinding(str(body.id), sha, names, tuple(lower), tuple(upper), root, profile)
    chains = tuple(chain_from_urdf(urdf, base_link=root, tip_link=link) for link, _ in frames)
    chain_coordinates = []
    for chain in chains:
        coordinates = []
        for name in chain.actuated_names:
            joint, scale, bias = kin.joints[name], 1., 0.
            while joint.mimic is not None:
                bias += scale * joint.mimic.offset
                scale *= joint.mimic.multiplier
                joint = kin.joints[joint.mimic.leader]
            if joint.name in passive:
                coordinates.append((-1, 0., bias + scale*passive[joint.name]))
            else:
                coordinates.append((names.index(joint.name), scale, bias))
        chain_coordinates.append(tuple(coordinates))
    return PreparedTarget(binding, kin, tuple(frames), tuple(i for i, name in enumerate(names) if name not in holds),
                          chains, tuple(chain_coordinates))


@dataclass(frozen=True)
class IKConfig:
    max_iterations: int = 100
    damping: float = 1e-3
    max_step_rad: float = 0.2
    position_tolerance_m: float = 0.002
    orientation_tolerance_rad: float = 0.01
    orientation_weight: float = 0.2

    def __post_init__(self):
        _require(type(self.max_iterations) is int and self.max_iterations > 0, "Positive iteration budget required")
        _require(all(math.isfinite(v) and v > 0 for name, v in asdict(self).items() if name != "max_iterations"),
                 "Solver settings must be finite and positive")


@dataclass(frozen=True)
class RetargetSequence:
    binding: TargetBinding
    config: IKConfig
    root_p: np.ndarray
    root_quat: np.ndarray
    q: np.ndarray
    fps: float
    position_residual_m: np.ndarray
    orientation_residual_rad: np.ndarray
    iterations: tuple[int, ...]
    dispositions: tuple[Disposition, ...]

    @property
    def feasible(self):
        return all(x is Disposition.GEOMETRICALLY_FEASIBLE for x in self.dispositions)

    def npz_arrays(self):
        _require(self.feasible, "Infeasible sequence cannot be exported as a training motion")
        return {"root_p": self.root_p, "root_quat": self.root_quat, "q": self.q,
                "fps": np.asarray(self.fps), "joint_names": np.asarray(self.binding.joint_names),
                "embodiment_id": np.asarray(self.binding.embodiment_id),
                "target_binding_sha256": np.asarray(self.binding.sha256)}

    def provenance(self):
        return {"scope": "geometric retarget only; no collision, balance or training admission",
                "target": asdict(self.binding), "target_binding_sha256": self.binding.sha256,
                "solver": asdict(self.config), "root_mode": "explicit_world_from_urdf_root",
                "dispositions": [x.value for x in self.dispositions], "iterations": list(self.iterations),
                "position_residual_m": self.position_residual_m.tolist(),
                "orientation_residual_rad": self.orientation_residual_rad.tolist()}


def _rotations(value, shape):
    array = np.array(value, dtype=np.float64, copy=True)
    _require(array.shape == shape and np.isfinite(array).all(), "Orientation array shape or values invalid")
    _require(np.allclose(np.swapaxes(array, -1, -2) @ array, np.eye(3), rtol=0, atol=1e-6)
             and np.allclose(np.linalg.det(array), 1, rtol=0, atol=1e-6), "Orientations must be proper rotations")
    return array


def solve_sequence(target: PreparedTarget, positions: Mapping[str, np.ndarray], *, root_p,
                   root_quat, fps: float, initial_q, orientations: Mapping[str, np.ndarray] | None = None,
                   config: IKConfig = IKConfig()) -> RetargetSequence:
    """Bounded multi-landmark IK, warm-started across every frame, on the CPU.

    An explicit root trajectory can translate and rotate freely in 3D. It is not
    optimized or guessed from source landmarks. Failures remain per-frame records;
    later frames are still solved and the complete sequence is never silently cut.
    """
    b = target.binding
    _require(math.isfinite(fps) and fps > 0, "Positive finite source FPS required")
    root_p = np.array(root_p, dtype=np.float64, copy=True)
    root_quat = np.array(root_quat, dtype=np.float64, copy=True)
    _require(root_p.ndim == 2 and root_p.shape[1] == 3 and len(root_p) >= 2 and np.isfinite(root_p).all(),
             "Root position must be finite (T,3), T>=2")
    T = len(root_p)
    _require((T-1)/fps >= 0.02, "Sequence must span at least one 50 Hz control step")
    _require(root_quat.shape == (T, 4) and np.isfinite(root_quat).all()
             and np.allclose(np.linalg.norm(root_quat, axis=1), 1, rtol=0, atol=1e-6),
             "Root quaternion must be unit wxyz (T,4)")
    required = {m.source for m in b.profile.landmarks}
    _require(set(positions) == required, "Source positions must match the explicit landmark mapping")
    points = {}
    for name, value in positions.items():
        array = np.array(value, dtype=np.float64, copy=True)
        _require(array.shape == (T, 3) and np.isfinite(array).all(), f"{name}: invalid source positions")
        points[name] = array
    required_orientations = {m.source for m in b.profile.landmarks if m.orientation}
    orientations = {} if orientations is None else orientations
    _require(set(orientations) == required_orientations, "Source orientations must match explicit orientation constraints")
    rotations = {name: _rotations(value, (T, 3, 3)) for name, value in orientations.items()}
    lo, hi = np.asarray(b.lower), np.asarray(b.upper)
    seed = np.array(initial_q, dtype=np.float64, copy=True)
    _require(seed.shape == lo.shape and np.isfinite(seed).all() and np.all(seed >= lo) and np.all(seed <= hi),
             "Explicit initial native state must be finite and within bounds")
    for name, value in b.profile.holds:
        _require(seed[b.joint_names.index(name)] == value, "Initial state disagrees with explicit hold")
    free = np.asarray(target.free_indices, dtype=int)
    q_out, p_out, r_out, dispositions, iterations = [], [], [], [], []
    for frame in range(T):
        def residual(q):
            rows, pe, re = [], [], []
            actual = target.poses(q, root_p[frame], root_quat[frame])
            for mapping, (p, R) in zip(b.profile.landmarks, actual, strict=True):
                dp = points[mapping.source][frame]-p
                pe.append(float(np.linalg.norm(dp)))
                rows.extend(dp * math.sqrt(mapping.weight))
                if mapping.orientation:
                    dr = orientation_error(mat_to_quat(R), mat_to_quat(rotations[mapping.source][frame]))
                    re.append(float(np.linalg.norm(dr)))
                    rows.extend(dr * config.orientation_weight * math.sqrt(mapping.weight))
                else:
                    re.append(0.0)  # No orientation target; profile records that absence.
            return np.asarray(rows), np.asarray(pe), np.asarray(re)
        q = seed.copy()
        used = 0
        for used in range(config.max_iterations+1):
            error, pe, re = residual(q)
            if (np.all(pe <= config.position_tolerance_m) and np.all(re <= config.orientation_tolerance_rad)) or not len(free) or used == config.max_iterations:
                break
            rows = []
            spatial = target.jacobians(q, root_quat[frame])
            actual = target.poses(q, root_p[frame], root_quat[frame])
            for mapping, full, (_, R) in zip(b.profile.landmarks, spatial, actual, strict=True):
                weight = math.sqrt(mapping.weight)
                rows.append(-full[:3, free] * weight)
                if mapping.orientation:
                    dr = orientation_error(mat_to_quat(R), mat_to_quat(rotations[mapping.source][frame]))
                    x, y, z = dr
                    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                    theta = np.linalg.norm(dr)
                    coefficient = 1/12 if theta < 1e-5 else (1 - theta/(2*math.tan(theta/2)))/(theta*theta)
                    right_inverse = np.eye(3) + .5*skew + coefficient*skew@skew
                    rows.append(-right_inverse @ full[3:, free] * config.orientation_weight * weight)
            jac = np.concatenate(rows)
            step = -np.linalg.solve(jac.T @ jac + config.damping**2*np.eye(len(free)), jac.T @ error)
            size = np.linalg.norm(step)
            if size > config.max_step_rad:
                step *= config.max_step_rad/size
            # Backtracking prevents a locally worse update from discarding the best pose.
            improved = False
            for scale in (1, 0.5, 0.25, 0.125, 0.0625):
                candidate = q.copy()
                candidate[free] = np.clip(q[free]+scale*step, lo[free], hi[free])
                if np.linalg.norm(residual(candidate)[0]) < np.linalg.norm(error):
                    q, improved = candidate, True
                    break
            if not improved:
                break
        _, pe, re = residual(q)  # Measure returned state, never a stale pre-update error.
        ok = np.all(pe <= config.position_tolerance_m) and np.all(re <= config.orientation_tolerance_rad)
        dispositions.append(Disposition.GEOMETRICALLY_FEASIBLE if ok else Disposition.RESIDUAL_EXCEEDED)
        q_out.append(q.copy()); p_out.append(pe); r_out.append(re); iterations.append(used)
        seed = q
    return RetargetSequence(b, config, _readonly(root_p), _readonly(root_quat), _readonly(q_out), float(fps),
                            _readonly(p_out), _readonly(r_out), tuple(iterations), tuple(dispositions))


@dataclass(frozen=True)
class MissingFact:
    code: str
    coordinate: str | None
    detail: str


@dataclass(frozen=True)
class TrainingPreflight:
    embodiment_id: str
    urdf_sha256: str
    native_joint_names: tuple[str, ...]
    simulation_assets: tuple[str, ...]
    missing_facts: tuple[MissingFact, ...]

    def to_dict(self):
        return {**asdict(self), "training_ready": False,
                "scope": "source preflight; a verified simulation/control profile is still required"}


def training_preflight(body: Embodiment) -> TrainingPreflight:
    """Report source gaps without repairing or asserting qualification of a body."""
    urdf = body.urdf_bytes
    kin = Kinematics.of(urdf)
    names = tuple(body.joint_names)
    missing = []
    for coordinate, name in zip(body.state.coordinates, names, strict=True):
        if isinstance(coordinate.axis.actuation, UndocumentedDrive):
            missing.append(MissingFact("undocumented_drive", name, "Owning embodiment has no documented drive relation"))
        if coordinate.axis.unit is not CoordinateUnit.RADIAN or not isinstance(coordinate.axis.bounds, Bounds):
            missing.append(MissingFact("unsupported_native_coordinate", name, "Current XMimic requires bounded angular coordinates"))
    for name in sorted(set(kin.independent)-set(names)):
        missing.append(MissingFact("unaccounted_description_coordinate", name, "Native state omits a movable URDF coordinate; explicit projection required"))
    assets = tuple(str(a.asset.content.sha256) for a in body.assets if a.asset.format is AssetFormat.MJCF)
    if not assets:
        missing.append(MissingFact("missing_mjcf", None, "Body declares no active MuJoCo description"))
    missing.extend((MissingFact("simulation_profile_required", None, "Floating root, actuator dynamics, default state and solver settings need a verified owner projection"),
                    MissingFact("contact_profile_required", None, "Support links, contact offsets and ground model need an explicit target profile")))
    return TrainingPreflight(str(body.id), hashlib.sha256(urdf).hexdigest(), names, assets, tuple(missing))
