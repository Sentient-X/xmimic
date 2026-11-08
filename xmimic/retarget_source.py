"""Bounded CPU decoding of BONES-SEED SOMA BVH into metric world-space poses.

SOMA translations are centimetres and rotations are degrees. Translation channels
replace a joint's OFFSET (including Hips); they are not displacement from OFFSET.
Rotations multiply in declared channel order. Every articulated joint, including
the virtual Root, participates in FK. No facing direction or root is inferred.

Conventions follow bones-studio/seed-viewer/soma/bvh.py and NVIDIA/soma-retargeter
at cbdb2b494a2af82137bf9b6f15122d60167281c4. This module does not fit robot joints,
infer contacts, retime motion, or admit a training example.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import math
import re

import numpy as np

CONVERTER_VERSION = "xmimic.soma-bvh/v1"
LENGTH_SCALE_M = 0.01
_AXES = {"X": 0, "Y": 1, "Z": 2}
_CHANNELS = {axis + kind for axis in _AXES for kind in ("position", "rotation")}


class SourceError(ValueError):
    """Source bytes, limits, or a frame transform are invalid; no clip is emitted."""


@dataclass(frozen=True)
class ParseLimits:
    """Explicit per-clip resource and numeric bounds; exceeding one is a refusal."""

    max_bytes: int = 64 * 1024**2
    max_header_bytes: int = 1024**2
    max_frames: int = 100_000
    max_joints: int = 256
    max_pose_samples: int = 2_000_000
    max_abs_length_m: float = 10_000.0
    max_abs_rotation_degrees: float = 1_000_000.0

    def __post_init__(self):
        for key in ("max_bytes", "max_header_bytes", "max_frames", "max_joints", "max_pose_samples"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise SourceError(f"{key} must be a positive integer")
        # Recursive hierarchy parsing is bounded below Python's recursion limit.
        if self.max_joints > 256:
            raise SourceError("max_joints cannot exceed 256")
        for key in ("max_abs_length_m", "max_abs_rotation_degrees"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise SourceError(f"{key} must be finite and positive")


@dataclass(frozen=True)
class Clip:
    """A complete human clip, retaining the source timebase and articulated Root.

    Positions are metres; rotations map each source joint's local axes into the
    explicitly supplied world frame. Arrays are float64 and read-only. End Site
    offsets are validated but are not invented as articulated joints.
    """

    joint_names: tuple[str, ...]
    parents: np.ndarray
    channels: tuple[tuple[str, ...], ...]
    offsets_m: np.ndarray
    world_positions: np.ndarray
    world_rotations: np.ndarray
    frame_time_s: float
    frame_time_literal: str
    source_sha256: str
    world_from_source: np.ndarray
    converter_version: str = CONVERTER_VERSION
    length_scale_m: float = LENGTH_SCALE_M
    rotation_unit: str = "degrees"

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time_s

    @property
    def frame_count(self) -> int:
        return len(self.world_positions)

    def joint_index(self, name: str) -> int:
        try:
            return self.joint_names.index(name)
        except ValueError as error:
            raise SourceError("Requested source joint is absent") from error


def _number(token: str) -> float:
    try:
        value = float(token)
    except ValueError as error:
        raise SourceError("Invalid numeric source field") from error
    if not math.isfinite(value):
        raise SourceError("Source contains a nonfinite number")
    return value


def _readonly(value, dtype=np.float64):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _placement(value, limits: ParseLimits) -> np.ndarray:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SourceError("world_from_source must be a rigid 4x4 transform") from error
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise SourceError("world_from_source must be a finite rigid 4x4 transform")
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-8, rtol=0)):
        raise SourceError("world_from_source must be proper rigid: no scale or reflection")
    if np.max(np.abs(transform[:3, 3])) > limits.max_abs_length_m:
        raise SourceError("World placement exceeds the length bound")
    return transform.copy()


def _hierarchy(text: str, limits: ParseLimits):
    tokens = iter(re.findall(r"[{}]|[^\s{}]+", text))
    names, parents, offsets, channels = [], [], [], []

    def take(expected=None):
        try:
            token = next(tokens)
        except StopIteration as error:
            raise SourceError("Truncated BVH hierarchy") from error
        if expected is not None and token != expected:
            raise SourceError("Unexpected BVH hierarchy token")
        return token

    def offset():
        take("OFFSET")
        values = np.array([_number(take()) for _ in range(3)]) * LENGTH_SCALE_M
        if np.max(np.abs(values)) > limits.max_abs_length_m:
            raise SourceError("Source offset exceeds the length bound")
        return values

    def joint(parent):
        name = take()
        if name in ("{", "}") or name in names:
            raise SourceError("Empty or duplicate joint name")
        if len(names) >= limits.max_joints:
            raise SourceError("Joint count exceeds the parse limit")
        index = len(names)
        names.append(name)
        parents.append(parent)
        take("{")
        offsets.append(offset())
        take("CHANNELS")
        count = take()
        if not re.fullmatch(r"[0-6]", count):
            raise SourceError("Unsupported channel count")
        order = tuple(take() for _ in range(int(count)))
        if len(set(order)) != len(order) or not set(order) <= _CHANNELS:
            raise SourceError("Unknown or duplicate channel")
        translations = tuple(x for x in order if x.endswith("position"))
        if translations and (set(translations) != {x + "position" for x in _AXES}
                             or order[:3] != translations):
            raise SourceError("SOMA requires complete XYZ translation channels before rotations")
        channels.append(order)
        ended = False
        while True:
            token = take()
            if token == "}":
                return
            if token == "JOINT":
                if ended:
                    raise SourceError("End Site must terminate a leaf")
                joint(index)
            elif token == "End":
                if ended or index in parents[index + 1:]:
                    raise SourceError("Duplicate or non-leaf End Site")
                take("Site")
                take("{")
                offset()
                take("}")
                ended = True
            else:
                raise SourceError("Unexpected BVH hierarchy token")

    take("HIERARCHY")
    take("ROOT")
    joint(-1)
    if next(tokens, None) is not None:
        raise SourceError("Trailing hierarchy tokens or multiple roots")
    if not sum(map(len, channels)):
        raise SourceError("Motion must declare at least one channel")
    return tuple(names), np.asarray(parents), np.asarray(offsets), tuple(channels)


def _axis_rotation(axis: int, angles: np.ndarray) -> np.ndarray:
    matrices = np.broadcast_to(np.eye(3), (len(angles), 3, 3)).copy()
    a, b = (axis + 1) % 3, (axis + 2) % 3
    c, s = np.cos(angles), np.sin(angles)
    matrices[:, a, a] = matrices[:, b, b] = c
    matrices[:, a, b], matrices[:, b, a] = -s, s
    return matrices


def parse_soma_bvh(data: bytes, *, world_from_source, limits: ParseLimits | None = None) -> Clip:
    """Decode one complete SOMA BVH with a caller-selected source-to-world placement.

    ``world_from_source`` is a proper rigid 4x4 transform, with metric translation.
    To obtain Z-up output, supply the verified source-axis conversion explicitly.
    No default forward-axis convention, discarded root, resampling or clipping is
    permitted. A malformed/truncated clip is refused as a whole.
    """
    limits = ParseLimits() if limits is None else limits
    if not isinstance(data, bytes) or not data or len(data) > limits.max_bytes:
        raise SourceError("BVH must be nonempty bytes within the parse limit")
    placement = _placement(world_from_source, limits)
    try:
        stream = io.StringIO(data.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise SourceError("BVH must be UTF-8 text") from error
    header, header_size = [], 0
    for line in stream:
        header_size += len(line.encode("utf-8"))
        if header_size > limits.max_header_bytes:
            raise SourceError("Hierarchy exceeds the header limit")
        if line.strip() == "MOTION":
            break
        header.append(line)
    else:
        raise SourceError("Missing MOTION section")
    names, parents, offsets, channels = _hierarchy("".join(header), limits)

    def next_line():
        for line in stream:
            if line.strip():
                return line.strip()
        raise SourceError("Truncated motion header")

    count_match = re.fullmatch(r"Frames:\s*([0-9]+)", next_line())
    if not count_match or len(count_match[1]) > 9:
        raise SourceError("Invalid frame count")
    count = int(count_match[1])
    if not 0 < count <= limits.max_frames or count * len(names) > limits.max_pose_samples:
        raise SourceError("Frame count or pose allocation exceeds the parse limit")
    time_match = re.fullmatch(r"Frame\s+Time:\s*(\S+)", next_line())
    if not time_match:
        raise SourceError("Missing Frame Time")
    literal = time_match[1]
    dt = _number(literal)
    if dt <= 0 or not math.isfinite(1 / dt) or not math.isfinite(count * dt):
        raise SourceError("Frame Time must define a finite positive timebase")
    width = sum(map(len, channels))
    values = np.empty((count, width), dtype=np.float64)
    index = 0
    for line in stream:
        if not line.strip():
            continue
        row = line.split()
        if index >= count or len(row) != width:
            raise SourceError("Motion rows disagree with declared frame/channel counts")
        values[index] = [_number(token) for token in row]
        index += 1
    if index != count:
        raise SourceError("Truncated motion rows")

    positions = np.empty((count, len(names), 3), dtype=np.float64)
    rotations = np.empty((count, len(names), 3, 3), dtype=np.float64)
    cursor = 0
    for j, order in enumerate(channels):
        local_p = np.broadcast_to(offsets[j], (count, 3)).copy()
        local_r = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
        for channel in order:
            numbers = values[:, cursor]
            cursor += 1
            axis = _AXES[channel[0]]
            if channel.endswith("position"):
                metric = numbers * LENGTH_SCALE_M
                if np.max(np.abs(metric)) > limits.max_abs_length_m:
                    raise SourceError("Source translation exceeds the length bound")
                local_p[:, axis] = metric
            else:
                if np.max(np.abs(numbers)) > limits.max_abs_rotation_degrees:
                    raise SourceError("Source rotation exceeds the angular bound")
                local_r = local_r @ _axis_rotation(axis, np.deg2rad(numbers))
        parent = parents[j]
        if parent < 0:
            positions[:, j] = local_p @ placement[:3, :3].T + placement[:3, 3]
            rotations[:, j] = placement[:3, :3] @ local_r
        else:
            positions[:, j] = positions[:, parent] + np.einsum("tij,tj->ti", rotations[:, parent], local_p)
            rotations[:, j] = rotations[:, parent] @ local_r
    if (not np.isfinite(positions).all() or not np.isfinite(rotations).all()
            or np.max(np.abs(positions)) > limits.max_abs_length_m):
        raise SourceError("Composed source poses exceed the numeric bounds")
    return Clip(names, _readonly(parents, np.int32), channels, _readonly(offsets),
                _readonly(positions), _readonly(rotations), dt, literal,
                hashlib.sha256(data).hexdigest(), _readonly(placement))
