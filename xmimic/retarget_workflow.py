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
from sx_actions import urr
from sx_actions.urr import motion_to_dict

from .retarget_motion import motion_from_clip, motion_positions, motion_orientations, motion_directions

from .artifact_cache import ArtifactCache, CacheIdentity, ClipDisposition, content_hash
from .retarget_source import CONVERTER_VERSION, parse_soma_bvh
from .retarget_target import IKConfig, TargetProfile, prepare_target, solve_sequence
from .retarget_quality import NominalQuality, qualify_motion
from .embodiment_bundle import verify_nominal_receipt
from . import retarget_source, retarget_target, retarget_progress, retarget_motion


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
    nominal_quality: NominalQuality | None = None

    def __post_init__(self):
        if self.nominal_quality is not None and not isinstance(self.nominal_quality, NominalQuality):
            raise ValueError('Explicit nominal quality contract required')
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
        if scaled & segmented or scaled | segmented != {m.source for m in self.target.landmarks if m.position}:
            raise ValueError("Each positional landmark needs exactly one scalar or segment proportion mapping")
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
                  on_checkpoint: Callable[[dict], None] | None = None,
                  robot_dir: Path | None = None) -> RetargetReceipt:
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
    nominal_scene = None
    if recipe.nominal_quality is not None:
        if robot_dir is None:
            raise ValueError('Nominal quality projection requires an owner robot bundle')
        owner_receipt = verify_nominal_receipt(robot_dir)
        if owner_receipt['embodiment_id'] != str(body.id):
            raise ValueError('Nominal bundle belongs to another body')
        nominal_scene = owner_receipt['scene_sha256']
    source_sha = hashlib.sha256(source).hexdigest()
    source_shape = shape_sha256 or content_hash({"kind": "embedded_bvh_skeleton", "source_sha256": source_sha})
    solver_sha = content_hash({
        "source": hashlib.sha256(Path(retarget_source.__file__).read_bytes()).hexdigest(),
        "target": hashlib.sha256(Path(retarget_target.__file__).read_bytes()).hexdigest(),
        "workflow": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "progress": hashlib.sha256(Path(retarget_progress.__file__).read_bytes()).hexdigest(),
        "motion_adapter": hashlib.sha256(Path(retarget_motion.__file__).read_bytes()).hexdigest(),
        "urr": hashlib.sha256(Path(urr.__file__).read_bytes()).hexdigest(),
        "numpy": np.__version__, "converter": CONVERTER_VERSION,
        "quality": hashlib.sha256(Path(__file__).with_name("retarget_quality.py").read_bytes()).hexdigest(),
        "nominal_scene": nominal_scene,
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
        # Preserve the complete observed source vocabulary. Role spelling is
        # explicit identity correspondence, never inferred robot anatomy.
        source_roles = {name: name for name in clip.joint_names}
        motion = motion_from_clip(clip, roles=source_roles, frame='soma_world', start_time_ns=0)
        points = motion_positions(motion, tuple(source_roles))
        rotations = motion_orientations(motion, tuple(source_roles))
        human_p = points[recipe.source_root]
        human_R = rotations[recipe.source_root]
        origin = human_p * recipe.root_translation_scale
        root_p = origin + np.einsum("tij,j->ti", human_R, recipe.root_offset_m)
        root_R = human_R @ _rotation(recipe.source_root_from_robot_root)
        root_q = np.asarray([mat_to_quat(rotation) for rotation in root_R])
        positions = {name: origin + scale * (points[name] - human_p)
                     for name, scale in recipe.landmark_scales}
        for mapping in recipe.segment_targets:
            point = root_p + np.einsum('tij,j->ti', root_R, mapping.root_anchor_m)
            for parent, child, length in mapping.segments:
                vector = points[child] - points[parent]
                norm = np.linalg.norm(vector, axis=-1, keepdims=True)
                if np.any(norm < 1e-10):
                    raise ValueError('Cannot normalize a zero-length source segment')
                point = point + length * vector / norm
            positions[mapping.source] = point
        directions = motion_directions(motion, {
            mapping.key: (mapping.source_parent, mapping.source_child) for mapping in recipe.target.directions})
        orientations = {name: rotations[name] @ _rotation(offset)
                        for name, offset in recipe.orientation_offsets}
        if not all(np.isfinite(a).all() for a in (root_p, root_R, *positions.values(), *orientations.values())):
            raise ValueError("Recipe scales overflowed source geometry")
        journal = entry.frame_progress(clip_id, execution_ref, total_frames=clip.frame_count,
                                       joint_count=len(target.binding.joint_names),
                                       landmark_count=len(target.binding.profile.landmarks),
                                       direction_count=len(target.binding.profile.directions))
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
                                  orientations=orientations, directions=directions, config=recipe.solver,
                                  resume=journal.frames, on_frame=committed, stop_requested=stop_requested)
        candidate_arrays = {
            "root_p": sequence.root_p, "root_quat": sequence.root_quat, "q": sequence.q,
            "fps": np.asarray(sequence.fps), "joint_names": np.asarray(target.binding.joint_names),
            "embodiment_id": np.asarray(str(body.id)),
            "target_binding_sha256": np.asarray(target.binding.sha256),
        }
        projected_arrays, quality_report = candidate_arrays, None
        accepted = sequence.feasible
        if recipe.nominal_quality is not None:
            projected_arrays, quality_report = qualify_motion(candidate_arrays, sequence.position_residual_m,
                sequence.orientation_residual_rad, robot_dir, recipe.nominal_quality)
            _direction_quality(quality_report, sequence.direction_residual_rad, recipe, clip.frame_count)
            accepted = quality_report['accepted']
        disposition = ClipDisposition.SUCCEEDED if accepted else ClipDisposition.REJECTED
        provenance = {"source_sha256": source_sha, "source_group": source_group,
                      "shape_sha256": source_shape, "frame_time_literal": clip.frame_time_literal,
                      "frame_count": clip.frame_count, "recipe": asdict(recipe),
                      "source_motion_id": str(motion.id), "source_roles": source_roles,
                      "source_time_origin": "relative_clip_start",
                      "geometric_result": sequence.provenance(), "nominal_quality_result": quality_report,
                      "optimizer_goal_passed": sequence.feasible, "training_admitted": False}
        # Private work lives under the caller's persistent cache, never an implicit /tmp cache.
        with tempfile.TemporaryDirectory(prefix="solver-", dir=entry.path) as scratch:
            directory = Path(scratch)
            candidate = directory / "candidate.npz"
            np.savez(candidate, **candidate_arrays)
            shared_motion = directory / 'source-motion.json'
            shared_motion.write_text(canonical_json(motion_to_dict(motion)))
            artifacts = {candidate.name: candidate, shared_motion.name: shared_motion}
            if accepted:
                motion = directory / "motion.npz"
                np.savez(motion, **projected_arrays)
                artifacts[motion.name] = motion
            report = directory / "retarget.json"
            report.write_text(canonical_json(provenance))
            artifacts[report.name] = report
            entry.settle_clip(clip_id, execution_ref, status=disposition,
                              artifacts=artifacts,
                              metadata={"source_group": source_group, "frames": clip.frame_count,
                                        "training_admitted": False},
                              reason="" if accepted else "One or more frames exceeded the explicit reference quality contract")
        return RetargetReceipt(identity.key, entry.complete(), disposition, False, entry.path)


def _direction_quality(report: dict, residuals, recipe: RetargetRecipe, frames: int) -> None:
    """Nominal time/floor projection cannot erase a failed pose-direction objective."""
    if not recipe.target.directions:
        return
    values = np.asarray(residuals, dtype=np.float64)
    if (values.shape != (frames, len(recipe.target.directions))
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError('Complete retained direction residuals required for nominal qualification')
    passed = bool(np.all(values <= recipe.solver.direction_tolerance_rad))
    report['checks']['direction_residual'] = passed
    report['direction_tolerance_rad'] = recipe.solver.direction_tolerance_rad
    report['max_direction_residual_rad'] = float(values.max())
    report['accepted'] = bool(report['accepted'] and passed)


def recipe_from_json(payload: bytes) -> RetargetRecipe:
    """One strict artifact boundary; job wire shapes remain platform protobufs."""
    from .retarget_target import Direction, Landmark, TargetKind

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
    if isinstance(raw, dict) and 'nominal_quality' not in raw:
        raw['nominal_quality'] = None  # Legacy recipes retain the optimizer-only gate.
    fields = set(RetargetRecipe.__dataclass_fields__)
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError("Retarget recipe fields differ from the supported artifact")
    target = raw["target"]
    if isinstance(target, dict) and 'directions' not in target:
        target['directions'] = []  # Explicit converter for pre-direction recipes.
    if not isinstance(target, dict) or set(target) != {"landmarks", "holds", "nominal_passive", "directions"}:
        raise ValueError("Explicit target mapping, holds and passive positions required")
    landmarks = []
    for item in target["landmarks"]:
        if isinstance(item, dict) and 'position' not in item:
            item['position'] = True
        if not isinstance(item, dict) or set(item) != {"source", "target", "kind", "orientation", "weight", "position"}:
            raise ValueError("Unsupported landmark fields")
        landmarks.append(Landmark(**{**item, "kind": TargetKind(item["kind"])}))
    directions = []
    for item in target['directions']:
        if isinstance(item, dict) and 'kind' not in item:
            item['kind'] = TargetKind.LINK.value
        if not isinstance(item, dict) or set(item) != set(Direction.__dataclass_fields__):
            raise ValueError('Unsupported direction fields')
        directions.append(Direction(**{**item, "kind": TargetKind(item["kind"])}))
    raw["target"] = TargetProfile(tuple(landmarks), tuple(target["holds"]), tuple(target["nominal_passive"]), tuple(directions))
    if isinstance(raw['solver'], dict) and not directions:
        # Older recipes had no directional objectives; these defaults cannot change their fit.
        raw['solver'].setdefault('direction_tolerance_rad', IKConfig().direction_tolerance_rad)
        raw['solver'].setdefault('direction_weight', IKConfig().direction_weight)
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
    for direction in target["directions"]:
        numeric(direction["weight"])
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
    quality = raw['nominal_quality']
    if quality is not None:
        if not isinstance(quality, dict) or set(quality) != set(NominalQuality.__dataclass_fields__):
            raise ValueError('Every nominal quality setting must be explicit')
        raw['nominal_quality'] = NominalQuality(**quality)
    raw["solver"] = IKConfig(**raw["solver"])
    return RetargetRecipe(**raw)


def qualify_cached_candidate(source_directory: Path, recipe: RetargetRecipe, robot_dir: Path,
                             *, cache: ArtifactCache, execution_ref: str) -> RetargetReceipt:
    """Apply a new explicit nominal contract to retained IK work without solving again.

    Mapping, optimizer configuration and original source are immutable. This creates
    a distinct derived cache containing the original candidate and report as well as
    the projected reference and new contract. It never changes the old rejection.
    """
    from dataclasses import replace
    source_directory = Path(source_directory)
    document = json.loads((source_directory / 'manifest.json').read_bytes())
    original_identity = CacheIdentity(**document['identity'])
    if source_directory.name != original_identity.key or len(document['clips']) != 1 or not document['complete']:
        raise ValueError('One complete retained source cache required')
    clip_id = next(iter(document['clips']))
    with ArtifactCache(source_directory.parent).entry(original_identity, [clip_id]) as original:
        artifacts = original.document['clips'][clip_id]['artifacts']
        if 'candidate.npz' not in artifacts or 'retarget.json' not in artifacts:
            raise ValueError('Retained unprojected candidate and optimizer report required')
        candidate_path = original.path / 'blobs' / artifacts['candidate.npz']['sha256']
        report_path = original.path / 'blobs' / artifacts['retarget.json']['sha256']
        original_report = json.loads(report_path.read_bytes())
        original_recipe = recipe_from_json(canonical_json(original_report['recipe']).encode())
        if recipe.nominal_quality is None or replace(recipe, nominal_quality=None) != replace(original_recipe, nominal_quality=None):
            raise ValueError('Requalification cannot change mapping or optimizer settings')
        if original_report['source_sha256'] != original_identity.source_sha256:
            raise ValueError('Source identity differs from retained optimizer report')
        owner = verify_nominal_receipt(robot_dir)
        if owner['embodiment_id'] != original_identity.embodiment_id or owner['urdf_sha256'] != original_identity.urdf_sha256:
            raise ValueError('Requalification cannot change canonical body facts')
        source_manifest_sha = hashlib.sha256((source_directory/'manifest.json').read_bytes()).hexdigest()
        identity = replace(original_identity,
            solver_sha256=content_hash({'original_solver': original_identity.solver_sha256,
                'quality_code': hashlib.sha256(Path(__file__).with_name('retarget_quality.py').read_bytes()).hexdigest(),
                'requalification_code': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}),
            settings_sha256=content_hash({'recipe': asdict(recipe), 'source_manifest_sha256': source_manifest_sha,
                                         'nominal_scene_sha256': owner['scene_sha256']}))
        with cache.entry(identity, [clip_id]) as entry:
            if entry.document['complete']:
                status = ClipDisposition(entry.document['clips'][clip_id]['status'])
                return RetargetReceipt(identity.key, entry.complete(), status, True, entry.path)
            with np.load(candidate_path, allow_pickle=False) as data:
                arrays = {name: data[name].copy() for name in data.files}
            if len(arrays['q']) != original_report['frame_count']:
                raise ValueError('Retained candidate frame count differs')
            geometric = original_report['geometric_result']
            projected, quality_report = qualify_motion(arrays, geometric['position_residual_m'],
                geometric['orientation_residual_rad'], robot_dir, recipe.nominal_quality)
            _direction_quality(quality_report, geometric.get('direction_residual_rad'), recipe, len(arrays['q']))
            disposition = ClipDisposition.SUCCEEDED if quality_report['accepted'] else ClipDisposition.REJECTED
            report = {**original_report, 'recipe': asdict(recipe), 'nominal_quality_result': quality_report,
                'optimizer_goal_passed': all(x == 'geometrically_feasible' for x in geometric['dispositions']),
                'retained_solver_manifest_sha256': source_manifest_sha, 'training_admitted': False}
            entry.start_clip(clip_id, execution_ref)
            with tempfile.TemporaryDirectory(prefix='qualification-', dir=entry.path) as temporary:
                directory = Path(temporary)
                output_report = directory/'retarget.json'
                output_report.write_text(canonical_json(report))
                output_files = {'candidate.npz': candidate_path, 'retarget.json': output_report,
                                'original-retarget.json': report_path}
                if 'source-motion.json' in artifacts:
                    output_files['source-motion.json'] = original.path / 'blobs' / artifacts['source-motion.json']['sha256']
                if quality_report['accepted']:
                    motion = directory/'motion.npz'
                    np.savez(motion, **projected)
                    output_files['motion.npz'] = motion
                entry.settle_clip(clip_id, execution_ref, status=disposition, artifacts=output_files,
                    metadata={'source_group': original_report['source_group'], 'frames': len(arrays['q']),
                              'training_admitted': False, 'retained_solver_manifest_sha256': source_manifest_sha},
                    reason='' if quality_report['accepted'] else 'Nominal reference quality contract exceeded')
            return RetargetReceipt(identity.key, entry.complete(), disposition, False, entry.path)
