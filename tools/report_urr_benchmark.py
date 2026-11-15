#!/usr/bin/env python3
"""Report independent per-role screening metrics without promoting tiny pilots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xmimic.artifact_cache import _atomic, _json
from xmimic.urr_benchmark import digest


def paired_interval(candidate, baseline, *, seed=7301, samples=10000, confidence=.95, comparisons=8):
    """Paired two-level bootstrap over training seeds and common task families.

    Screening intervals remain descriptive when three training seeds are used.
    No step within a trajectory is counted as an independent replicate.
    """
    candidate, baseline = np.asarray(candidate, float), np.asarray(baseline, float)
    if (candidate.shape != baseline.shape or candidate.ndim != 2 or min(candidate.shape) < 1
            or not np.isfinite(candidate).all() or not np.isfinite(baseline).all()):
        raise ValueError('Paired finite training-seed-by-task measurements required')
    delta = candidate-baseline
    rng = np.random.default_rng(seed)
    seed_indices = rng.integers(0, delta.shape[0], (samples, delta.shape[0]))
    task_indices = rng.integers(0, delta.shape[1], (samples, delta.shape[1]))
    means = delta[seed_indices[:, :, None], task_indices[:, None, :]].mean((1, 2))
    return {'mean_candidate_minus_baseline': float(delta.mean()),
            'upper_bound': float(np.quantile(means, 1-(1-confidence)/comparisons)),
            'training_seeds': delta.shape[0], 'task_families': delta.shape[1],
            'confidence': confidence, 'familywise_comparisons': comparisons,
            'method': 'paired training-seed/task-family bootstrap; descriptive screening only'}


def report(run: Path, output: Path):
    run, output = Path(run), Path(output)
    result = json.loads((run/'result.json').read_bytes())
    request = json.loads((run/'request.json').read_bytes())
    if not result['complete']:
        raise ValueError('Cannot summarize an incomplete screening as complete')
    by_method, table = {}, []
    for record in result['records']:
        method, seed = record['method'], record['training_seed']
        task_files = [run/method/f'seed-{seed}'/'evaluation'/f'task-{s}.json' for s in request['evaluation_seeds']]
        tasks = [json.loads(p.read_bytes()) for p in task_files]
        metric = {}
        for key in ('position_mm','orientation_deg','aperture_mm'):
            values = np.asarray([[r[key] for r in task['measurements'][-request['task']['hold_steps']:]] for task in tasks])
            metric[key] = np.quantile(values,.95,axis=1)
        metric['held_failure'] = np.asarray([float(not t['held']) for t in tasks])
        metric['constraint_failure'] = np.asarray([float(t['constraint_failed']) for t in tasks])
        metric['target_identities'] = [t['targets'] for t in tasks]
        by_method.setdefault(method,{})[seed] = metric
        row = {'method':method,'training_seed':seed,'held_tasks':sum(t['held'] for t in tasks),'tasks':len(tasks),
               'constraint_failed_tasks':sum(t['constraint_failed'] for t in tasks)}
        for key in ('position_mm','orientation_deg','aperture_mm'):
            row[key+'_mean_terminal_p95_by_role'] = metric[key].mean(0).tolist()
        table.append(row)
    comparisons = []
    for control in ('ppo','world_model','ppo_coarse'):
        if control not in by_method or 'combined' not in by_method:
            continue
        seeds = sorted(by_method[control].keys() & by_method['combined'].keys())
        if not seeds:
            continue
        for seed in seeds:
            if by_method[control][seed]['target_identities'] != by_method['combined'][seed]['target_identities']:
                raise ValueError('Cannot compare methods evaluated on different URR targets')
        for metric in ('position_mm','orientation_deg','aperture_mm','held_failure','constraint_failure'):
            a = np.asarray([by_method['combined'][s][metric] for s in seeds])
            b = np.asarray([by_method[control][s][metric] for s in seeds])
            if a.ndim == 2:
                a,b = a[...,None],b[...,None]
            for role in range(a.shape[-1]):
                comparisons.append({'control':control,'metric':metric,'role_index':role,
                                    **paired_interval(a[...,role],b[...,role])})
    summary = {'schema':'xmimic.urr-screening-report/v1','run_result_sha256':digest(run/'result.json'),
               'scope':'Actual nominal-simulator screening. Adapted learners; finite small budget, not native reproduction or physical qualification.',
               'table':table,'paired_descriptive_intervals':comparisons,
               'decision':'do_not_promote' if any(r['held_tasks'] < r['tasks'] or r['constraint_failed_tasks'] for r in table if r['method']=='combined') else 'confirmation_required',
               'native_reproduction':'pending_external_native_job_results',
               'interpretation':'This budget does not establish converged learned performance or absence of regression. No architecture is selected from aggregate reward.'}
    if 'raw' in output.resolve().parts or output.is_symlink():
        raise ValueError('Report must be a derived regular artifact')
    _atomic(output, _json(summary))
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=report(args.run,args.output)
    print(json.dumps({'decision':result['decision'],'output':str(args.output),'rows':len(result['table'])}))
