"""Authored CPU retarget and single-GPU training programs for sx_training.

Entrypoints: ``xmimic.managed_worker:retarget`` and ``:train``. Inputs are already
verified context.artifacts; this module has no Catalog credentials or cloud SDK.
CPU input names: embodiment.json, source.bvh, recipe.json, training-profile.json,
assets/<canonical package-relative URDF>, robot/<owner nominal export files>.
CPU outputs are transform candidates. Catalog alone admits a feasible selection
for the GPU entrypoint; geometric success never changes false eligibility to true.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

from sx_contracts import Sha256Digest
from sx_contracts.identity import canonical_json
from sx_embodiments import Embodiment

from .artifact_cache import ArtifactCache, ClipDisposition
from .checkpoint import RecoveryOptions, run_binding
from .config import Config
from .embodiment_bundle import TrainingProfile, adapt_nominal_bundle, verify_nominal_receipt
from .retarget_source import ParseLimits
from .retarget_workflow import recipe_from_json, retarget_clip


MAX_OUTPUT_BYTES = 1024**3


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json(payload):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('Duplicate program artifact key')
            out[key] = value
        return out
    def reject(_):
        raise ValueError('Nonfinite program artifact value')
    return json.loads(payload, object_pairs_hook=pairs, parse_constant=reject)


def _read(artifacts, name, limit):
    path = artifacts[name]
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('Input artifact exceeds its supported bound')
    return path.read_bytes()


@contextmanager
def _staged_file(destination, staging=None):
    """Admit complete derived bytes atomically; a kill leaves only disposable staging."""
    destination = Path(destination)
    staging = Path(staging) if staging is not None else destination.parent / '.materializing'
    if 'raw' in destination.resolve().parts or destination.is_symlink() or staging.is_symlink():
        raise ValueError('Materialization requires regular derived paths')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='pending-', dir=staging)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        yield temporary
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)  # exclusive, never overwrite admitted bytes
        except FileExistsError:
            if destination.is_symlink() or _digest(destination) != _digest(temporary):
                raise ValueError('Retained output conflicts with current input') from None
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _copy(source, destination, *, staging=None):
    """Idempotent materialization, never replace retained different bytes."""
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_file() or destination.is_symlink():
        raise ValueError('Program artifacts must be regular files')
    if destination.exists():
        if _digest(source) != _digest(destination):
            raise ValueError('Retained output conflicts with current input')
        return
    with _staged_file(destination, staging) as temporary:
        with source.open('rb') as inp, temporary.open('wb') as out:
            shutil.copyfileobj(inp, out, 1024**2)


def _write_bytes(path, encoded, *, staging=None):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Retained program artifact differs')
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError('Retained program artifact differs')
        return
    with _staged_file(path, staging) as temporary:
        temporary.write_bytes(encoded)


def _write(path, value, *, staging=None):
    _write_bytes(path, canonical_json(value).encode(), staging=staging)


def _copy_prefix(artifacts, prefix, directory, *, staging=None):
    found = False
    for name, path in artifacts.items():
        if name.startswith(prefix + '/'):
            relative = Path(name.removeprefix(prefix + '/'))
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Input path escapes its bundle')
            _copy(path, directory / relative, staging=staging)
            found = True
    if not found:
        raise ValueError(f'Missing {prefix} input bundle')


@contextmanager
def _body(artifacts, directory, *, staging=None):
    body = Embodiment.from_json(_read(artifacts, 'embodiment.json', 8*1024**2).decode())
    _copy_prefix(artifacts, 'assets', directory, staging=staging)
    previous = os.environ.get('SX_EMBODIMENTS_ASSETS')
    os.environ['SX_EMBODIMENTS_ASSETS'] = str(directory)
    try:
        # Resolution and digest checking belong to the embodiment owner.
        _ = body.urdf_bytes
        yield body
    finally:
        if previous is None:
            os.environ.pop('SX_EMBODIMENTS_ASSETS', None)
        else:
            os.environ['SX_EMBODIMENTS_ASSETS'] = previous


def _manifest(directory, *, purpose, eligible, filename="artifact-manifest.json", selection=None, staging=None):
    from sx_training.artifacts import ArtifactFile, ArtifactManifest
    manifest = ArtifactManifest(purpose, tuple(
        ArtifactFile(p.relative_to(directory).as_posix(), Sha256Digest(_digest(p)), p.stat().st_size)
        for p in sorted(directory.rglob('*')) if p.is_file() and p.name not in {filename, 'training-receipt.json'}
        and (selection is None or p.relative_to(directory).as_posix() in selection)
    ), eligible)
    if any(f.size_bytes > MAX_OUTPUT_BYTES for f in manifest.files):
        raise ValueError('Output exceeds Catalog per-object limit; choose bounded source/compute shards')
    destination = directory / filename
    data = manifest.canonical_bytes()
    _write_bytes(destination, data, staging=staging)


def retarget(context) -> Path:
    """CPU job: preserve source family and whole-clip disposition; return candidate pack."""
    from sx_training.artifacts import ArtifactPurpose
    expected = {'embodiment_id', 'clip_id', 'source_group', 'shape_sha256'}
    if set(context.config) != expected:
        raise ValueError('CPU program needs exact embodiment, clip, family and shape bindings')
    artifacts = context.artifacts
    inherited = context.artifact_manifest.training_eligible
    if context.artifact_manifest.purpose is not ArtifactPurpose.TRANSFORM:
        raise ValueError('Retarget input must be an admitted transform artifact')
    recipe = recipe_from_json(_read(artifacts, 'recipe.json', 1024**2))
    raw_profile = _json(_read(artifacts, 'training-profile.json', 1024**2))
    if not isinstance(raw_profile, dict) or set(raw_profile) != {f.name for f in fields(TrainingProfile)}:
        raise ValueError('Explicit complete training profile required')
    profile = TrainingProfile(**raw_profile)
    work = context.run_dir / 'xmimic'
    staging = work / 'materializing'
    output = context.run_dir / 'artifact'
    robot = output / 'robot'
    from .retarget_progress import verify_progress_checkpoint
    from .retarget_target import RetargetInterrupted
    cache = ArtifactCache(work / 'retarget-cache')
    def restore_progress(path):
        verify_progress_checkpoint(cache, _json(path.read_bytes()), clip_id=context.config['clip_id'],
                                   execution_ref=context.job_id)
    context.restore(restore_progress)
    context.recovery_complete()
    latest_progress = []
    def save_progress(receipt, *, force=False):
        latest_progress[:] = [receipt]
        completed = receipt['completed']
        if force or completed % 32 == 0 or completed == receipt['total_frames'] or context.stopping:
            path = work / 'progress-receipts' / f'frame-{completed:09d}.json'
            _write(path, receipt, staging=staging)
            context.checkpoint(path, step=completed)
            context.log({"retarget/frames": completed, "retarget/total_frames": receipt["total_frames"]}, step=completed)
    _copy_prefix(artifacts, 'robot', robot, staging=staging)
    with _body(artifacts, work / 'owner-assets', staging=staging) as body:
        if str(body.id) != context.config['embodiment_id']:
            raise ValueError('Worker canonical body identity differs')
        receipt = verify_nominal_receipt(robot)
        if receipt['embodiment_id'] != str(body.id) or receipt['urdf_sha256'] != hashlib.sha256(body.urdf_bytes).hexdigest():
            raise ValueError('Nominal simulation export does not describe the retarget body')
        if not (robot / 'meta.json').exists():
            adapt_nominal_bundle(robot, profile)
        else:
            from sx_contracts.identity import content_digest
            from .model import load_model
            _, meta = load_model(robot, motion_format='retargeted_npz')
            if (meta['training_profile_sha256'] != str(content_digest(asdict(profile)))
                    or meta['owner_receipt_sha256'] != _digest(robot/'receipt.json')
                    or meta['owner_receipt'] != receipt):
                raise ValueError('Retained simulation training profile differs')
        try:
            result = retarget_clip(body, _read(artifacts, 'source.bvh', ParseLimits().max_bytes), recipe,
                                   expected_embodiment_id=str(body.id), cache=cache,
                                   clip_id=context.config['clip_id'], source_group=context.config['source_group'],
                                   execution_ref=context.job_id, shape_sha256=context.config['shape_sha256'],
                                   stop_requested=lambda: context.stopping, on_checkpoint=save_progress,
                                   robot_dir=robot)
        except RetargetInterrupted:
            if latest_progress:
                save_progress(latest_progress[-1], force=True)
            context.interrupted()
            raise
    document = _json((result.directory / 'manifest.json').read_bytes())
    clip = document['clips'][context.config['clip_id']]
    for name, reference in clip['artifacts'].items():
        target = output / ('motion/clip.npz' if name == 'motion.npz' else 'diagnostics/' + name)
        _copy(result.directory / 'blobs' / reference['sha256'], target, staging=staging)
    _copy(result.directory / 'manifest.json', output / 'retarget-cache.json', staging=staging)
    _write(output / 'candidate.json', {
        'schema': 'xmimic.retarget-candidate/v1', 'cache_key': result.cache_key,
        'cache_manifest_sha256': result.manifest_sha256, 'embodiment_id': context.config['embodiment_id'],
        'source_group': context.config['source_group'], 'disposition': str(result.disposition),
        'training_admitted': False, 'inherited_training_eligible': inherited,
        'feasible': result.disposition is ClipDisposition.SUCCEEDED,
    }, staging=staging)
    _copy(artifacts['recipe.json'], output / 'recipe.json', staging=staging)
    _copy(artifacts['training-profile.json'], output / 'training-profile.json', staging=staging)
    if result.disposition is ClipDisposition.SUCCEEDED:
        selection = {p.relative_to(output).as_posix() for p in robot.rglob('*') if p.is_file()}
        selection |= {'motion/clip.npz', 'candidate.json', 'retarget-cache.json', 'recipe.json', 'training-profile.json'}
        _manifest(output, purpose=ArtifactPurpose.TRANSFORM, eligible=inherited,
                  filename='training-candidate.json', selection=selection, staging=staging)
    _manifest(output, purpose=ArtifactPurpose.TRANSFORM, eligible=inherited, staging=staging)
    return output


def _pack_state(directory, destination):
    if destination.exists():
        with tarfile.open(destination, 'r:') as archive:
            members = archive.getmembers()
            files = {p.name: p for p in directory.iterdir() if p.is_file()}
            if (len(members) != len(files) or {m.name for m in members} != files.keys()
                    or any(not m.isfile() for m in members)):
                raise ValueError('Retained native archive membership differs')
            for member in members:
                with archive.extractfile(member) as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if digest != _digest(files[member.name]):
                    raise ValueError('Retained native archive bytes differ')
        return
    with _staged_file(destination) as temporary:
        with tarfile.open(temporary, 'w:') as archive:
            for path in sorted(directory.iterdir()):
                if path.is_symlink() or not path.is_file():
                    raise ValueError('Native checkpoint contains non-file member')
                archive.add(path, arcname=path.name, recursive=False)


def _unpack_state(archive, directory, expected_binding):
    """Read only named numeric checkpoint members; never tar.extractall."""
    directory.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, 'r:') as bundle:
        members = bundle.getmembers()
        if any(not m.isfile() for m in members) or len({m.name for m in members}) != len(members):
            raise ValueError('Unsafe private checkpoint archive')
        names = {m.name for m in members}
        if 'manifest.json' not in names or bundle.getmember('manifest.json').size > 16*1024**2:
            raise ValueError('Private checkpoint manifest missing or oversized')
        stream = bundle.extractfile('manifest.json')
        manifest_bytes = stream.read()
        manifest = _json(manifest_bytes)
        if manifest.get('schema') != 'xmimic.training-state/v1' or manifest['binding'] != expected_binding:
            raise ValueError('Private checkpoint binding differs')
        records = manifest['leaves']
        expected = {f'{i:06d}.npy' for i in range(len(records))} | {'manifest.json'}
        if names != expected:
            raise ValueError('Private checkpoint membership differs')
        (directory / 'manifest.json').write_bytes(manifest_bytes)
        for i, record in enumerate(records):
            name = f'{i:06d}.npy'
            if record['file'] != name or bundle.getmember(name).size != record['size']:
                raise ValueError('Private checkpoint member size differs')
            source = bundle.extractfile(name)
            with source, (directory / name).open('xb') as target:
                shutil.copyfileobj(source, target, 1024**2)
            if _digest(directory / name) != record['sha256']:
                raise ValueError('Private checkpoint member hash differs')
    return directory


def train(context) -> Path:
    """GPU job: consume Catalog-admitted feasible motions and preserve complete state."""
    from sx_training.artifacts import ArtifactPurpose
    from .single_gpu import run
    if set(context.config) != {'training', 'source_binding', 'embodiment_id'}:
        raise ValueError('GPU program requires training config, source and embodiment bindings')
    if context.artifact_manifest.purpose is not ArtifactPurpose.TRAINING or not context.artifact_manifest.training_eligible:
        raise ValueError('GPU input lacks eligible training admission')
    artifacts = context.artifacts
    candidate = _json(_read(artifacts, 'candidate.json', 1024**2))
    if candidate['feasible'] is not True or candidate['disposition'] != 'succeeded':
        raise ValueError('Rejected retarget output cannot train')
    if candidate['embodiment_id'] != context.config['embodiment_id']:
        raise ValueError('GPU motion/body binding differs')
    if (candidate['cache_manifest_sha256'] != context.config['source_binding']
            or _digest(artifacts['retarget-cache.json']) != context.config['source_binding']):
        raise ValueError('GPU source selection differs from the admitted candidate')
    cache_document = _json(_read(artifacts, 'retarget-cache.json', 8*1024**2))
    clips = list(cache_document['clips'].values())
    if (cache_document['complete'] is not True or len(clips) != 1 or clips[0]['status'] != 'succeeded'
            or cache_document['identity']['embodiment_id'] != context.config['embodiment_id']
            or clips[0]['artifacts']['motion.npz']['sha256'] != _digest(artifacts['motion/clip.npz'])):
        raise ValueError('GPU motion bytes differ from the completed retarget record')
    work = context.run_dir / 'xmimic'
    staging = work / 'materializing'
    robot, motion = work / 'robot', work / 'motion'
    _copy_prefix(artifacts, 'robot', robot, staging=staging)
    _copy_prefix(artifacts, 'motion', motion, staging=staging)
    receipt = verify_nominal_receipt(robot)
    meta = _json((robot/'meta.json').read_bytes())
    if (meta['embodiment_id'] != context.config['embodiment_id']
            or meta['owner_receipt_sha256'] != _digest(robot/'receipt.json')
            or meta['owner_receipt'] != receipt):
        raise ValueError('GPU nominal body receipt differs')
    raw = context.config['training']
    controlled = {'robot_dir', 'motion_dir', 'log_dir', 'motion_format', 'layout'}
    allowed = {f.name for f in fields(Config)} - controlled
    if not isinstance(raw, dict) or not set(raw) <= allowed:
        raise ValueError('Unsupported or infrastructure-controlled training option')
    defaults = Config()
    for name, value in raw.items():
        if type(value) is not type(getattr(defaults, name)):
            raise ValueError('Training option type differs from Config')
    cfg = Config(**raw, robot_dir=str(robot), motion_dir=str(motion), log_dir=str(work / 'logs'),
                 motion_format='retargeted_npz', layout='1GPU')
    if cfg.wm_mirror > 0 and 'mirror_permutation' not in meta:
        raise ValueError('Training requires wm_mirror=0 without an explicit body symmetry map')
    states = work / 'states'
    states.mkdir(parents=True, exist_ok=True)
    binding = run_binding(cfg, context.config['source_binding'])
    resume = []
    def accept_archive(path):
        destination = Path(tempfile.mkdtemp(prefix='restore-', dir=work)) / 'state'
        resume.append(_unpack_state(path, destination, binding))
    context.restore(accept_archive)
    # An admitted native boundary may be ahead of the platform envelope if a process
    # stopped between the two writes. Verify its binding before choosing it.
    for path in sorted(states.glob('state_*')):
        if path.is_dir() and (path / 'manifest.json').is_file():
            manifest = _json((path / 'manifest.json').read_bytes())
            if manifest['binding'] != binding:
                raise ValueError('Retained native training state has another binding')
            resume.append(path)
    resume_from = max(resume, key=lambda p: _json((p/'manifest.json').read_bytes())['iteration']) if resume else None
    last = []
    def checkpoint(path, iteration):
        archive = work / f'native-state-{iteration:09d}.tar'
        _pack_state(path, archive)
        context.checkpoint(archive, step=iteration)
        last[:] = [path]
    def restored():
        context.recovery_complete()
        if resume_from is not None:
            last[:] = [resume_from]
            if resume_from.parent == states:
                checkpoint(resume_from, _json((resume_from/'manifest.json').read_bytes())['iteration'])
    run(cfg, recovery=RecoveryOptions(str(states), str(resume_from) if resume_from else None,
                                      source_binding=context.config['source_binding'], on_checkpoint=checkpoint,
                                      on_restored=restored, stop_requested=lambda: context.stopping,
                                      handle_signals=False,
                                      on_metrics=lambda metrics, step: context.log(metrics, step=step)))
    if context.stopping:
        context.interrupted()
    output = context.run_dir / 'artifact'
    output.mkdir(exist_ok=True)
    policies = sorted((work / 'logs').glob('ckpt_*.npz'))
    if not policies:
        raise ValueError('Training did not produce a deployable policy artifact')
    _copy(policies[-1], output / 'policy.npz', staging=staging)
    for name in ('train_log.csv', 'compile.json', 'args.json'):
        _copy(work / 'logs' / name, output / 'metrics' / name, staging=staging)
    final = last[-1] if last else resume_from
    if final is None:
        raise ValueError('Training completed without recoverable state')
    final_archive = work / 'final-native-state.tar'
    _pack_state(final, final_archive)
    export_recovery_archive(final_archive, output/'recovery', staging=staging)
    _write(output / 'run.json', {'schema': 'xmimic.managed-training/v1', 'binding': binding,
                               'iteration': _json((final/'manifest.json').read_bytes())['iteration'],
                               'requested_iterations': cfg.iterations,
                               'physical_qualification': False}, staging=staging)
    return output


def export_recovery_archive(archive: Path, destination: Path, *, chunk_bytes=MAX_OUTPUT_BYTES, staging=None) -> Path:
    """Preserve the final opaque state in bounded Catalog objects, with byte-order identity.

    Intermediate snapshots remain under the managed runtime. This final export is
    separate from its automatic checkpoint reattachment mechanism.
    """
    if type(chunk_bytes) is not int or not 0 < chunk_bytes <= MAX_OUTPUT_BYTES:
        raise ValueError('Invalid recovery chunk bound')
    archive, destination = Path(archive), Path(destination)
    if archive.is_symlink() or not archive.is_file() or 'raw' in destination.resolve().parts:
        raise ValueError('Recovery export requires regular derived paths')
    destination.mkdir(parents=True, exist_ok=True)
    staging = Path(staging) if staging is not None else destination.parent / '.materializing'
    staging.mkdir(parents=True, exist_ok=True)
    whole, offset, records = hashlib.sha256(), 0, []
    with archive.open('rb') as source:
        while offset < archive.stat().st_size:
            name = f'part-{len(records):06d}.bin'
            target = destination/name
            descriptor, temporary = tempfile.mkstemp(prefix='.partial-', dir=staging)
            sha, size = hashlib.sha256(), 0
            with os.fdopen(descriptor, 'wb') as output:
                while size < chunk_bytes:
                    data = source.read(min(1024**2, chunk_bytes-size))
                    if not data:
                        break
                    output.write(data)
                    sha.update(data)
                    whole.update(data)
                    size += len(data)
                output.flush()
                os.fsync(output.fileno())
            if size == 0:
                raise ValueError('Recovery source changed during export')
            if target.exists():
                if target.is_symlink() or target.stat().st_size != size or _digest(target) != sha.hexdigest():
                    raise ValueError('Retained recovery chunk differs')
            else:
                os.link(temporary, target)  # exclusive admission; never replace previous paid bytes
            Path(temporary).unlink()
            records.append({'file':name,'offset':offset,'size_bytes':size,'sha256':sha.hexdigest()})
            offset += size
    if not records or offset != archive.stat().st_size:
        raise ValueError('Empty or changed recovery archive')
    index = destination/'index.json'
    _write(index, {'schema':'xmimic.recovery-chunks/v1','size_bytes':offset,'sha256':whole.hexdigest(),
                   'chunk_bytes':chunk_bytes,'chunks':records}, staging=staging)
    return index


def restore_recovery_archive(directory: Path, destination: Path) -> Path:
    """Verify a downloaded final Catalog closure and reconstruct its opaque tar.

    This does not resume a trainer or acknowledge a platform checkpoint. The result
    still requires native checkpoint binding and template validation before training.
    """
    directory, destination = Path(directory), Path(destination)
    index_path = directory/'index.json'
    if index_path.is_symlink() or index_path.stat().st_size > 1024**2 or 'raw' in destination.resolve().parts:
        raise ValueError('Unsafe recovery restore path or index')
    index = _json(index_path.read_bytes())
    if set(index) != {'schema','size_bytes','sha256','chunk_bytes','chunks'} or index['schema'] != 'xmimic.recovery-chunks/v1':
        raise ValueError('Unsupported recovery index')
    if type(index['chunk_bytes']) is not int or not 0 < index['chunk_bytes'] <= MAX_OUTPUT_BYTES:
        raise ValueError('Invalid recovery chunk bound')
    Sha256Digest(index['sha256'])
    if type(index['size_bytes']) is not int or index['size_bytes'] <= 0 or not index['chunks']:
        raise ValueError('Invalid recovery archive size')
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.recovery-', dir=destination.parent)
    whole, offset = hashlib.sha256(), 0
    with os.fdopen(descriptor, 'wb') as output:
        for number, record in enumerate(index['chunks']):
            if (set(record) != {'file','offset','size_bytes','sha256'}
                    or record['file'] != f'part-{number:06d}.bin' or type(record['offset']) is not int
                    or record['offset'] != offset or type(record['size_bytes']) is not int
                    or not 0 < record['size_bytes'] <= index['chunk_bytes']
                    or (number < len(index['chunks'])-1 and record['size_bytes'] != index['chunk_bytes'])):
                raise ValueError('Recovery chunk membership or byte order differs')
            Sha256Digest(record['sha256'])
            source = directory/record['file']
            if source.is_symlink() or not source.is_file() or source.stat().st_size != record['size_bytes']:
                raise ValueError('Recovery chunk size differs')
            sha = hashlib.sha256()
            with source.open('rb') as stream:
                for data in iter(lambda: stream.read(1024**2), b''):
                    output.write(data)
                    sha.update(data)
                    whole.update(data)
            if sha.hexdigest() != record['sha256']:
                raise ValueError('Recovery chunk digest differs')
            offset += record['size_bytes']
        output.flush()
        os.fsync(output.fileno())
    if offset != index['size_bytes'] or whole.hexdigest() != index['sha256']:
        raise ValueError('Complete recovery archive identity differs')
    if destination.exists():
        if destination.is_symlink() or destination.stat().st_size != offset or _digest(destination) != whole.hexdigest():
            raise ValueError('Existing recovered archive differs')
    else:
        os.link(temporary, destination)
    Path(temporary).unlink()
    return destination
