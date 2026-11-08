"""SOMA → canonical target → immutable CPU cache, with explicit geometric assumptions.

The same function runs locally or inside an admitted CPU Job. It buys no compute,
does not choose a robot by tensor width, and never substitutes a failed retarget.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from sx_contracts.identity import canonical_json
from sx_geometry import mat_to_quat

from .artifact_cache import ArtifactCache, CacheIdentity, ClipDisposition, content_hash
from .retarget_source import CONVERTER_VERSION, parse_soma_bvh
from .retarget_target import IKConfig, TargetProfile, prepare_target, solve_sequence
from . import retarget_source, retarget_target, retarget_progress


@dataclass(frozen=True)
class SegmentTarget:
    """Explicit robot-root anchor and source segment directions at target lengths."""
    source: str
    root_anchor_m: tuple[float, float, float]
    segments: tuple[tuple[str, str, float], ...]

    def __post_init__(self):
        object.__setattr__(self, 'root_anchor_m', tuple(float(v) for v in self.root_anchor_m))
        object.__setattr__(self, 'segments', tuple((a,b,float(length)) for a,b,length in self.segments))
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError('Segment target needs a source landmark name')
        if len(self.root_anchor_m) != 3 or not np.isfinite(self.root_anchor_m).all():
            raise ValueError('Segment anchor must be a finite metric vector')
        if not self.segments or any(not isinstance(a,str) or not a or not isinstance(b,str) or not b
                                    or a == b or not math.isfinite(length) or length <= 0
                                    for a,b,length in self.segments):
            raise ValueError('Segment chain needs explicit endpoints and positive target lengths')


@dataclass(frozen=True)
class RetargetRecipe:
    """Target mapping and source-to-robot transforms, kept separate from hardware facts.

    Root offset is expressed in the human root frame. Each landmark is scaled
    relative to the human root; it is not scaled about the world origin. Root
    displacement and landmark proportions can therefore have different scales.
    """

    target: TargetProfile
    world_from_source: tuple[tuple[float, ...], ...]
    source_root: str
    source_root_from_robot_root: tuple[tuple[float, ...], ...]
    root_offset_m: tuple[float, float, float]
    root_translation_scale: float
    landmark_scales: tuple[tuple[str, float], ...]
    orientation_offsets: tuple[tuple[str, tuple[tuple[float, ...], ...]], ...]
    initial_q: tuple[tuple[str, float], ...]
    solver: IKConfig
    segment_targets: tuple[SegmentTarget, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'segment_targets', tuple(self.segment_targets))
        if (not all(isinstance(t, SegmentTarget) for t in self.segment_targets)
                or len({t.source for t in self.segment_targets}) != len(self.segment_targets)):
            raise ValueError('Duplicate or invalid segment targets')
        # Freeze nested values; a caller cannot change an already-keyed recipe.
        for field in ("world_from_source", "source_root_from_robot_root"):
            object.__setattr__(self, field, tuple(tuple(float(v) for v in row) for row in getattr(self, field)))
        object.__setattr__(self, "root_offset_m", tuple(float(v) for v in self.root_offset_m))
        object.__setattr__(self, "landmark_scales", tuple((n, float(v)) for n, v in self.landmark_scales))
        object.__setattr__(self, "initial_q", tuple((n, float(v)) for n, v in self.initial_q))
        object.__setattr__(self, "orientation_offsets", tuple(
            (n, tuple(tuple(float(v) for v in row) for row in rotation)) for n, rotation in self.orientation_offsets))
        if not isinstance(self.source_root, str) or not self.source_root.strip() or not math.isfinite(self.root_translation_scale) or self.root_translation_scale <= 0:
            raise ValueError("An explicit root and positive translation scale are required")
        world = np.asarray(self.world_from_source)
        if world.shape != (4, 4) or not np.isfinite(world).all() or not np.array_equal(world[3], [0, 0, 0, 1]):
            raise ValueError("world_from_source must be an explicit rigid 4x4 transform")
        _rotation(world[:3, :3])
        _rotation(self.source_root_from_robot_root)
        if len(self.root_offset_m) != 3 or not np.isfinite(self.root_offset_m).all():
            raise ValueError("Root offset must be a finite metric three-vector")
        for values in (self.landmark_scales, self.orientation_offsets, self.initial_q):
            if any(not isinstance(n, str) or not n.strip() for n, _ in values) or len(dict(values)) != len(values):
                raise ValueError("Recipe contains duplicate names")
        scaled = set(dict(self.landmark_scales))
        segmented = {t.source for t in self.segment_targets}
        if scaled & segmented or scaled | segmented != {m.source for m in self.target.landmarks}:
            raise ValueError("Each landmark needs exactly one scalar or segment proportion mapping")
        if not all(math.isfinite(v) and v > 0 for _, v in self.landmark_scales):
            raise ValueError("Landmark scales must be positive and finite")
        if set(dict(self.orientation_offsets)) != {m.source for m in self.target.landmarks if m.orientation}:
            raise ValueError("Every orientation target needs an explicit axis offset")
        for _, rotation in self.orientation_offsets:
            _rotation(rotation)
        if not all(math.isfinite(v) for _, v in self.initial_q):
            raise ValueError("Initial coordinates must be finite")


def _rotation(value):
    matrix = np.asarray(value, dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(matrix), 1, atol=1e-8, rtol=0)):
        raise ValueError("Orientation offset must be a proper 3x3 rotation")
    return matrix


@dataclass(frozen=True)
class RetargetReceipt:
    cache_key: str
    manifest_sha256: str
    disposition: ClipDisposition
    cache_hit: bool
    directory: Path


def retarget_clip(body, source: bytes, recipe: RetargetRecipe, *, expected_embodiment_id: str,
                  cache: ArtifactCache, clip_id: str, source_group: str,
                  execution_ref: str, shape_sha256: str | None = None,
                  stop_requested: Callable[[], bool] = lambda: False,
                  on_progress: Callable[[int, int], None] | None = None,
                  on_checkpoint: Callable[[dict], None] | None = None) -> RetargetReceipt:
    """Account for every source frame; only a fully feasible clip gets motion.npz.

    A rejected clip still retains its candidate coordinates and residuals. Neither
    output is collision/balance qualification or Catalog training admission.
    """
    if any(not isinstance(v, str) or not v.strip() for v in (clip_id, source_group, execution_ref)):
        raise ValueError("Source-family and execution identities are required")
    target = prepare_target(body, recipe.target, expected_embodiment_id=expected_embodiment_id)
    initial = dict(recipe.initial_q)
    if set(initial) != set(target.binding.joint_names):
        raise ValueError("Initial state must name every canonical coordinate exactly once")
    seed = np.asarray([initial[n] for n in target.binding.joint_names])
    if np.any(seed < target.binding.lower) or np.any(seed > target.binding.upper):
        raise ValueError("Initial state outside canonical bounds")
    if any(initial[n] != value for n, value in recipe.target.holds):
        raise ValueError("Initial state differs from explicit held coordinates")
    source_sha = hashlib.sha256(source).hexdigest()
    source_shape = shape_sha256 or content_hash({"kind": "embedded_bvh_skeleton", "source_sha256": source_sha})
    solver_sha = content_hash({
        "source": hashlib.sha256(Path(retarget_source.__file__).read_bytes()).hexdigest(),
        "target": hashlib.sha256(Path(retarget_target.__file__).read_bytes()).hexdigest(),
        "workflow": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "progress": hashlib.sha256(Path(retarget_progress.__file__).read_bytes()).hexdigest(),
        "numpy": np.__version__, "converter": CONVERTER_VERSION,
    })
    identity = CacheIdentity(source_sha, source_shape, str(body.id), target.binding.urdf_sha256,
                             target.binding.sha256, solver_sha,
                             content_hash({"recipe": asdict(recipe), "source_group": source_group, "clip_id": clip_id}))
    with cache.entry(identity, [clip_id]) as entry:
        state = entry.document["clips"][clip_id]
        if entry.document["complete"] or state["status"] in {"succeeded", "rejected", "failed"}:
            return RetargetReceipt(identity.key, entry.complete(), ClipDisposition(state["status"]), True, entry.path)
        # Parsing/validation occurs before recording paid solver work as started.
        clip = parse_soma_bvh(source, world_from_source=recipe.world_from_source)
        if clip.frame_count < 2 or (clip.frame_count - 1) * clip.frame_time_s < .02:
            raise ValueError("Source clip must span at least one 50 Hz control step")
        root_index = clip.joint_index(recipe.source_root)
        human_p = clip.world_positions[:, root_index]
        human_R = clip.world_rotations[:, root_index]
        origin = human_p * recipe.root_translation_scale
        root_p = origin + np.einsum("tij,j->ti", human_R, recipe.root_offset_m)
        root_R = human_R @ _rotation(recipe.source_root_from_robot_root)
        root_q = np.asarray([mat_to_quat(rotation) for rotation in root_R])
        positions = {name: origin + scale * (clip.world_positions[:, clip.joint_index(name)] - human_p)
                     for name, scale in recipe.landmark_scales}
        for mapping in recipe.segment_targets:
            point = root_p + np.einsum('tij,j->ti', root_R, mapping.root_anchor_m)
            for parent, child, length in mapping.segments:
                vector = clip.world_positions[:,clip.joint_index(child)] - clip.world_positions[:,clip.joint_index(parent)]
                norm = np.linalg.norm(vector, axis=-1, keepdims=True)
                if np.any(norm < 1e-10):
                    raise ValueError('Cannot normalize a zero-length source segment')
                point = point + length * vector / norm
            positions[mapping.source] = point
        orientations = {name: clip.world_rotations[:, clip.joint_index(name)] @ _rotation(offset)
                        for name, offset in recipe.orientation_offsets}
        if not all(np.isfinite(a).all() for a in (root_p, root_R, *positions.values(), *orientations.values())):
            raise ValueError("Recipe scales overflowed source geometry")
        journal = entry.frame_progress(clip_id, execution_ref, total_frames=clip.frame_count,
                                       joint_count=len(target.binding.joint_names),
                                       landmark_count=len(target.binding.profile.landmarks))
        if on_checkpoint is not None:
            on_checkpoint(journal.checkpoint_receipt())
        def committed(frame):
            journal.append(frame)
            if on_checkpoint is not None:
                on_checkpoint(journal.checkpoint_receipt())
            if on_progress is not None:
                on_progress(frame.index + 1, clip.frame_count)
        sequence = solve_sequence(target, positions, root_p=root_p, root_quat=root_q,
                                  fps=clip.fps, initial_q=[initial[n] for n in target.binding.joint_names],
                                  orientations=orientations, config=recipe.solver,
                                  resume=journal.frames, on_frame=committed, stop_requested=stop_requested)
        disposition = ClipDisposition.SUCCEEDED if sequence.feasible else ClipDisposition.REJECTED
        provenance = {"source_sha256": source_sha, "source_group": source_group,
                      "shape_sha256": source_shape, "frame_time_literal": clip.frame_time_literal,
                      "frame_count": clip.frame_count, "recipe": asdict(recipe),
                      "geometric_result": sequence.provenance(), "training_admitted": False}
        # Private work lives under the caller's persistent cache, never an implicit /tmp cache.
        with tempfile.TemporaryDirectory(prefix="solver-", dir=entry.path) as scratch:
            directory = Path(scratch)
            motion = directory / ("motion.npz" if sequence.feasible else "candidate.npz")
            arrays = sequence.npz_arrays() if sequence.feasible else {
                "root_p": sequence.root_p, "root_quat": sequence.root_quat, "q": sequence.q,
                "fps": np.asarray(sequence.fps), "joint_names": np.asarray(target.binding.joint_names),
                "embodiment_id": np.asarray(str(body.id)),
            }
            np.savez(motion, **arrays)
            report = directory / "retarget.json"
            report.write_text(canonical_json(provenance))
            entry.settle_clip(clip_id, execution_ref, status=disposition,
                              artifacts={motion.name: motion, report.name: report},
                              metadata={"source_group": source_group, "frames": clip.frame_count,
                                        "training_admitted": False},
                              reason="" if sequence.feasible else "One or more frames exceeded geometric residual limits")
        return RetargetReceipt(identity.key, entry.complete(), disposition, False, entry.path)


def recipe_from_json(payload: bytes) -> RetargetRecipe:
    """One strict artifact boundary; job wire shapes remain platform protobufs."""
    from .retarget_target import Landmark, TargetKind

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate recipe key")
            result[key] = value
        return result
    def invalid_constant(_):
        raise ValueError("Nonfinite recipe value")
    raw = json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid_constant)
    fields = set(RetargetRecipe.__dataclass_fields__)
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError("Retarget recipe fields differ from the supported artifact")
    target = raw["target"]
    if not isinstance(target, dict) or set(target) != {"landmarks", "holds", "nominal_passive"}:
        raise ValueError("Explicit target mapping, holds and passive positions required")
    landmarks = []
    for item in target["landmarks"]:
        if not isinstance(item, dict) or set(item) != {"source", "target", "kind", "orientation", "weight"}:
            raise ValueError("Unsupported landmark fields")
        landmarks.append(Landmark(**{**item, "kind": TargetKind(item["kind"])}))
    raw["target"] = TargetProfile(tuple(landmarks), tuple(target["holds"]), tuple(target["nominal_passive"]))
    if not isinstance(raw["solver"], dict) or set(raw["solver"]) != set(IKConfig.__dataclass_fields__):
        raise ValueError("Every solver setting must be explicit")
    # Do not coerce booleans/numeric strings at the artifact boundary.
    def numeric(value):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Recipe numeric fields must be finite JSON numbers")
    numeric(raw["root_translation_scale"])
    for field in ("world_from_source", "source_root_from_robot_root"):
        for row in raw[field]:
            for value in row:
                numeric(value)
    for value in raw["root_offset_m"]:
        numeric(value)
    for field in ("landmark_scales", "initial_q"):
        for _, value in raw[field]:
            numeric(value)
    for _, matrix in raw["orientation_offsets"]:
        for row in matrix:
            for value in row:
                numeric(value)
    for field in ("holds", "nominal_passive"):
        for _, value in target[field]:
            numeric(value)
    for landmark in target["landmarks"]:
        numeric(landmark["weight"])
    for value in raw["solver"].values():
        numeric(value)
    segment_targets = []
    for mapping in raw['segment_targets']:
        if not isinstance(mapping, dict) or set(mapping) != {'source','root_anchor_m','segments'}:
            raise ValueError('Unsupported segment target fields')
        for value in mapping['root_anchor_m']:
            numeric(value)
        for _, _, value in mapping['segments']:
            numeric(value)
        segment_targets.append(SegmentTarget(**mapping))
    raw['segment_targets'] = tuple(segment_targets)
    raw["solver"] = IKConfig(**raw["solver"])
    return RetargetRecipe(**raw)
