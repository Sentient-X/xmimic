"""Retain completed screening evidence privately; never admit it for training.

Uses the existing Catalog publisher and verified pack restore. Source files and
older checkpoints are read-only. Archive inventories retain empty source members.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

from sx_contracts import Sha256Digest
from sx_contracts.identity import canonical_json
from sx_training.artifacts import ArtifactFile, ArtifactManifest, ArtifactPurpose
from xmimic.artifact_cache import CatalogPublisher, Provenance, content_hash
from xmimic.managed_pipeline import restore_pack


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def files_under(root):
    root = Path(root)
    if any(p.is_symlink() for p in (root, *root.parents)) or not root.is_dir():
        raise ValueError('Evidence must be an existing regular directory')
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('Evidence contains a link or special file')
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path
    return result


def exact_file(path, payload):
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError('Retained publication identity changed')
    else:
        with path.open('xb') as stream:
            stream.write(payload)


def archive_files(files, destination):
    """Deterministic tar.gz plus each original logical member's exact identity."""
    inventory = {}
    for name, path in sorted(files.items()):
        posix = PurePosixPath(name)
        if posix.is_absolute() or '..' in posix.parts or str(posix) != name or path.is_symlink():
            raise ValueError('Unsafe archive member')
        inventory[name] = {'sha256':digest(path), 'size_bytes':path.stat().st_size}
    if not destination.exists():
        partial = destination.with_suffix('.partial')
        if not partial.exists():
            with partial.open('xb') as stream, gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode='w') as archive:
                    for name, path in sorted(files.items()):
                        member = tarfile.TarInfo(name)
                        member.size = inventory[name]['size_bytes']
                        member.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                        with path.open('rb') as source:
                            archive.addfile(member, source)
        # Recover a completed prior staging file only after verifying all bytes.
        # link is an atomic no-clobber seal even if another writer wins the race.
        verify_archive(partial, inventory)
        try:
            os.link(partial, destination)
        except FileExistsError:
            verify_archive(destination, inventory)
        partial.unlink()
    verify_archive(destination, inventory)
    return inventory


def verify_archive(path, inventory):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Archive must be a regular file')
    observed = {}
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            if not member.isfile() or member.name in observed or member.name not in inventory:
                raise ValueError('Unexpected or unsafe archived evidence member')
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError('Unreadable archived evidence')
            observed[member.name] = {'sha256':hashlib.file_digest(stream,'sha256').hexdigest(), 'size_bytes':member.size}
    if observed != inventory:
        raise ValueError('Archived evidence inventory differs')


def checkpoint_files(cell, training):
    checkpoint = cell / f"checkpoint-{training['iterations']:06d}"
    if not training['completed'] or Path(training['checkpoint']).resolve() != checkpoint.resolve():
        raise ValueError('Training receipt does not bind its final full checkpoint')
    candidates = sorted(cell.glob('checkpoint-*'))
    if not candidates or candidates[-1] != checkpoint:
        raise ValueError('The retained final checkpoint is not the latest')
    manifest = json.loads((checkpoint/'manifest.json').read_bytes())
    if manifest['schema'] != 'xmimic.training-state/v1' or manifest['iteration'] != training['iterations']:
        raise ValueError('Incomplete or wrong checkpoint boundary')
    files = files_under(checkpoint)
    if set(files) != {'manifest.json', *(leaf['file'] for leaf in manifest['leaves'])}:
        raise ValueError('Checkpoint membership differs from complete manifest')
    for leaf in manifest['leaves']:
        path = files[leaf['file']]
        if digest(path) != leaf['sha256'] or path.stat().st_size != leaf['size']:
            raise ValueError('Checkpoint leaf bytes changed')
    return {f'{checkpoint.name}/{name}': path for name,path in files.items()}


def prepare(screening, prepared, owner, output):
    screening, prepared, owner, output = map(Path, (screening, prepared, owner, output))
    if 'raw' in output.resolve().parts or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('Publication requires a derived regular directory')
    output.mkdir(parents=True, exist_ok=True)
    request = json.loads((screening/'request.json').read_bytes())
    result = json.loads((screening/'result.json').read_bytes())
    if result['complete'] is not True:
        raise ValueError('Only completed screening evidence can be published')
    report = json.loads((screening/'screening-report.json').read_bytes())
    if result['request_sha256'] != content_hash(request) or report['run_result_sha256'] != digest(screening/'result.json'):
        raise ValueError('Screening request/result/report linkage differs')
    expected_cells = {(method, seed) for method in request['methods']
                      for seed in (request['seeds'][:1] if method == 'dls' else request['seeds'])}
    actual_cells = [(record['method'], record['training_seed']) for record in result['records']]
    if not expected_cells or len(actual_cells) != len(expected_cells) or set(actual_cells) != expected_cells:
        raise ValueError('Screening does not contain exactly the requested method/seed cells')
    if digest(prepared/'embodiment.json') != request['body_sha256'] or digest(prepared/'benchmark.json') != request['preparation_sha256']:
        raise ValueError('Prepared body differs from completed screening')
    prepared_files = files_under(prepared)
    spec = json.loads((prepared/'benchmark.json').read_bytes())
    required = {'robot/scene.xml': spec['scene_sha256'], 'robot/canonical.urdf': spec['urdf_sha256'],
                'robot/receipt.json': spec['owner_receipt_sha256'], 'robot/embodiment.json': request['body_sha256']}
    owner_receipt = json.loads((prepared/'robot/receipt.json').read_bytes())
    required.update({f'robot/{name}': sha for name, sha in owner_receipt['files']})
    required.update({member['path']: member['sha256'] for member in spec['assets']})
    if any(name not in prepared_files or digest(prepared_files[name]) != sha for name, sha in required.items()):
        raise ValueError('Prepared scene/asset closure differs')
    if any(prepared_files[member['path']].stat().st_size != member['size_bytes'] for member in spec['assets']):
        raise ValueError('Prepared asset size differs')
    checkout = files_under(screening/'checkout')
    receipt = json.loads((screening/'checkout-receipt.json').read_bytes())
    checkout_hashes = {name:digest(path) for name,path in checkout.items()}
    checkout_extras = set(checkout) - receipt['files'].keys()
    if (any(checkout_hashes.get(name) != sha for name,sha in receipt['files'].items())
            or any('__pycache__' not in Path(name).parts or not name.endswith('.pyc') for name in checkout_extras)):
        raise ValueError('Historical checkout differs from retained source receipt')
    for field, name in (('runner_sha256','tools/benchmark_urr.py'),
                        ('environment_sha256','xmimic/urr_benchmark.py'),
                        ('learner_sha256','xmimic/urr_learning.py')):
        if checkout_hashes.get(name) != request[field]:
            raise ValueError('Historical source differs from executed benchmark code')
    evidence = {name:screening/name for name in ('request.json','result.json','screening-report.json','checkout-receipt.json')}
    evidence.update({f'source/{name}':path for name,path in files_under(screening/'source').items()})
    checkpoints = {}
    selected = []
    for record in result['records']:
        cell_name = f"{record['method']}/seed-{record['training_seed']}"
        cell = screening/cell_name
        evaluation = files_under(cell/'evaluation')
        expected_tasks = {f'task-{seed}.json' for seed in request['evaluation_seeds']}
        if set(evaluation) != expected_tasks | {'request.json','summary.json'}:
            raise ValueError('Evaluation task closure is incomplete or contains unrecorded files')
        if json.loads(evaluation['summary.json'].read_bytes()) != record['evaluation']:
            raise ValueError('Screening and evaluation receipts disagree')
        evaluation_request = json.loads(evaluation['request.json'].read_bytes())
        evaluation_hash = content_hash(evaluation_request)
        if evaluation_request['seeds'] != request['evaluation_seeds'] or record['evaluation']['request_sha256'] != evaluation_hash:
            raise ValueError('Evaluation belongs to another request')
        environment = evaluation_request['environment']
        if any(environment[field] != expected for field, expected in (
                ('body_document_sha256',request['body_sha256']),
                ('preparation_sha256',request['preparation_sha256']),
                ('scene_sha256',spec['scene_sha256']),
                ('environment_code_sha256',request['environment_sha256']))):
            raise ValueError('Evaluation belongs to another body, scene or code')
        for seed in request['evaluation_seeds']:
            task = json.loads(evaluation[f'task-{seed}.json'].read_bytes())
            if (task['request_sha256'] != evaluation_hash or task['seed'] != seed
                    or len(task['measurements']) != request['task']['episode_steps']):
                raise ValueError('Evaluation task identity or sample count differs')
        evidence.update({f'{cell_name}/evaluation/{name}':path for name,path in evaluation.items()})
        training_path = cell/'training.json'
        if record['method'] != 'dls':
            if digest(training_path) != record['training_receipt_sha256']:
                raise ValueError('Training receipt changed')
            training = json.loads(training_path.read_bytes())
            evidence[f'{cell_name}/training.json'] = training_path
            checkpoints.update({f'{cell_name}/{name}':path for name,path in checkpoint_files(cell,training).items()})
            selected.append(f"{cell_name}/checkpoint-{training['iterations']:06d}")
    owner_files = files_under(owner/'sx_embodiments')
    owner_files = {f'sx_embodiments/{name}':path for name,path in owner_files.items() if '__pycache__' not in Path(name).parts}
    owner_files.update({name:owner/name for name in ('README.md','LICENSE','THIRD_PARTY_NOTICES.md','pyproject.toml')})
    owner_files['owner-commit.txt'] = owner.parent/'commit.txt'
    groups = {'prepared':files_under(prepared), 'screening':evidence,
              'final-checkpoints':checkpoints, 'historical-checkout':checkout,
              'owner-source':owner_files, 'visual-review-task-0-v3':files_under(screening/'visual-review-task-0-v3')}
    inventories = {}
    named = {}
    for name, files in groups.items():
        archive = output/f'{name}.tar.gz'
        inventories[archive.name] = archive_files(files, archive)
        named[archive.name] = archive
    inventory = {'schema':'xmimic.urr-evidence-inventory/v1','purpose':'evaluation','training_eligible':False,
        'screening_revision':receipt['revision'], 'owner_revision':(owner.parent/'commit.txt').read_text().strip(),
        'embodiment_id':json.loads((prepared/'benchmark.json').read_bytes())['embodiment_id'],
        'selected_complete_checkpoints':selected,'archives':inventories,
        'historical_checkout_runtime_byproducts':sorted(checkout_extras),
        'scope':'Private completed synthetic screening evidence; no learning admission or physical qualification. All earlier checkpoints remain local. Owner source excludes unrelated robot assets; complete prepared asset closure is retained. Original licenses remain in source and asset provenance.'}
    exact_file(output/'inventory.json',canonical_json(inventory).encode())
    named['inventory.json'] = output/'inventory.json'
    files = tuple(ArtifactFile(name,Sha256Digest(digest(path)),path.stat().st_size) for name,path in sorted(named.items()))
    if any(not 0 < f.size_bytes <= 1024**3 for f in files):
        raise ValueError('Archive requires sharding before Catalog publication')
    manifest = ArtifactManifest(ArtifactPurpose.EVALUATION,files,False)
    exact_file(output/'artifact-manifest.json',manifest.canonical_bytes())
    named['artifact-manifest.json'] = output/'artifact-manifest.json'
    return named, inventory


def publish(catalog, project, pack, output, named, inventory):
    manifest_path = output/'artifact-manifest.json'
    provenance = Provenance('https://github.com/Sentient-X/xmimic',inventory['screening_revision'],
                            'NOASSERTION','tools.publish_urr_benchmark')
    binding = CatalogPublisher(catalog,project,provenance).publish_files(
        {digest(path):path for path in named.values()},manifest_path,output/'transport',pack)
    restored = restore_pack(catalog,pack,digest(manifest_path),output/'readback')
    if restored.manifest.purpose is not ArtifactPurpose.EVALUATION or restored.manifest.training_eligible:
        raise ValueError('Readback purpose differs')
    restored_inventory = json.loads((restored.directory/'inventory.json').read_bytes())
    if restored_inventory != inventory:
        raise ValueError('Readback logical inventory differs')
    for name, members in inventory['archives'].items():
        verify_archive(restored.directory/name,members)
    receipt = {'schema':'xmimic.urr-evidence-publication/v1',**binding,
               'verified_readback':True,'logical_files':sum(map(len,inventory['archives'].values())),
               'published_bytes':sum(path.stat().st_size for path in named.values()),
               'training_eligible':False,'purpose':'evaluation','restored_directory':str(restored.directory)}
    exact_file(output/'publication-receipt.json',canonical_json(receipt).encode())
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('screening','prepared','owner','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--project',required=True)
    parser.add_argument('--pack',required=True)
    parser.add_argument('--publish',action='store_true')
    args = parser.parse_args()
    named, inventory = prepare(args.screening,args.prepared,args.owner,args.output)
    if not args.publish:
        print(json.dumps({'prepared_manifest':digest(args.output/'artifact-manifest.json'),'files':len(named),'bytes':sum(p.stat().st_size for p in named.values())}))
        return
    from sx.credentials import api_key_for, environment_endpoints
    from sx._transport import ConnectionProfile
    from sx_auth.principal import ProjectId
    from sx_catalog_client import CatalogClient
    from sx_service.registry import CATALOG
    profile = ConnectionProfile.PRODUCTION
    key = os.environ.get('SX_API_KEY') or api_key_for(profile)
    if not key:
        raise ValueError('Existing production authentication is unavailable')
    with CatalogClient(environment_endpoints(profile).for_door(CATALOG.name),api_key=key,project_id=ProjectId(args.project)) as catalog:
        receipt = publish(catalog,args.project,args.pack,args.output,named,inventory)
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
