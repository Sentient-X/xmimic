"""SOMA observations in the shared, body-independent URR Motion contract.

The existing parser owns BVH decoding, centimetres-to-metres and world placement.
This adapter only names observed roles and timestamps them. Robot proportions,
anatomical axis calibration, controls and training admission belong downstream.
Source bytes, their hash, correspondence and literal timebase remain provenance
beside Motion (e.g. in ReferenceMotion); they are not a second motion schema.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray
from sx_actions.urr import Motion, MotionError, MotionFrame
from sx_geometry import Pose, mat_to_quat, quat_to_mat

from .retarget_source import Clip, ParseLimits, parse_soma_bvh


def motion_from_clip(
    clip: Clip,
    *,
    roles: Mapping[str, str],
    frame: str,
    start_time_ns: int,
) -> Motion:
    """Preserve selected observed BVH poses in the clip's explicit world frame.

    ``roles`` maps shared role names to source BVH joint names, without inferring
    anatomy. ``frame`` names the frame already chosen by ``world_from_source``.
    ``start_time_ns`` is caller-supplied, not an invented capture epoch. Each time
    is rounded independently to the nearest nanosecond (ties to even) from the
    literal BVH period, so fractional-nanosecond periods do not accumulate drift.
    """
    if not isinstance(frame, str) or not frame.strip():
        raise MotionError("name the clip's world reference frame")
    if type(start_time_ns) is not int or start_time_ns < 0:
        raise MotionError("start time must be explicit non-negative integer nanoseconds")
    correspondence = dict(roles)
    if not correspondence or any(
        not isinstance(role, str) or not role.strip()
        or not isinstance(joint, str) or not joint.strip()
        for role, joint in correspondence.items()
    ):
        raise MotionError("provide explicit nonempty role-to-BVH-joint correspondence")
    indices = {role: clip.joint_index(joint) for role, joint in correspondence.items()}
    period_ns = Fraction(clip.frame_time_literal) * 1_000_000_000
    times = [start_time_ns + round(index * period_ns) for index in range(clip.frame_count)]
    if period_ns <= 0 or any(a >= b for a, b in zip(times, times[1:])):
        raise MotionError("source samples are not distinct at nanosecond resolution")
    return Motion(tuple(
        MotionFrame(
            time,
            frame,
            {role: Pose(clip.world_positions[index, joint],
                        mat_to_quat(clip.world_rotations[index, joint]))
             for role, joint in indices.items()},
        )
        for index, time in enumerate(times)
    ))


def soma_motion(
    source: bytes,
    *,
    roles: Mapping[str, str],
    frame: str,
    start_time_ns: int,
    world_from_source: object,
    limits: ParseLimits = ParseLimits(),
) -> Motion:
    """Decode complete SOMA bytes once, then retain their observed shared Motion."""
    clip = parse_soma_bvh(source, world_from_source=world_from_source, limits=limits)
    return motion_from_clip(clip, roles=roles, frame=frame, start_time_ns=start_time_ns)


def _roles(roles: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(roles)
    if any(not isinstance(role, str) or not role.strip() for role in selected) or len(set(selected)) != len(selected):
        raise MotionError("select distinct named observed roles")
    return selected


def _readonly(values: object) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def motion_positions(motion: Motion, roles: Sequence[str]) -> dict[str, NDArray[np.float64]]:
    """Select observed metric positions; missing samples never become zero targets."""
    selected = _roles(roles)
    if any(role not in sample.poses for sample in motion.frames for role in selected):
        raise MotionError("requested role lacks a position sample")
    return {role: _readonly([sample.poses[role].position for sample in motion.frames])
            for role in selected}


def motion_orientations(motion: Motion, roles: Sequence[str]) -> dict[str, NDArray[np.float64]]:
    """Select observed world rotation matrices; Position has no identity rotation."""
    result = {}
    for role in _roles(roles):
        values = []
        for sample in motion.frames:
            pose = sample.poses.get(role)
            if not isinstance(pose, Pose):
                raise MotionError("requested role lacks an orientation sample")
            values.append(quat_to_mat(pose.quaternion))
        result[role] = _readonly(values)
    return result


def motion_directions(
    motion: Motion, pairs: Mapping[str, tuple[str, str]],
) -> dict[str, NDArray[np.float64]]:
    """Unit directions from explicit role pairs, in Motion's unchanged world frame."""
    result = {}
    for key, pair in pairs.items():
        if not isinstance(key, str) or not key.strip() or len(pair) != 2 or pair[0] == pair[1]:
            raise MotionError("directions need a name and two distinct observed roles")
        parent, child = pair
        points = motion_positions(motion, (parent, child))
        vector = points[child] - points[parent]
        lengths = np.linalg.norm(vector, axis=1, keepdims=True)
        if np.any(lengths <= 1e-10):
            raise MotionError("coincident observed roles have no direction")
        result[key] = _readonly(vector / lengths)
    return result
