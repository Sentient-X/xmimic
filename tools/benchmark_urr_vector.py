#!/usr/bin/env python3
"""Measure actual CPU physics collection throughput; never score a policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from xmimic.urr_benchmark import TaskConfig, URRPhysicsEnv


def benchmark(prepared, num_envs, steps):
    config = TaskConfig(precision=True, accumulate=True, episode_steps=steps+8, hold_steps=3)
    scalar = URRPhysicsEnv(prepared, config)
    scalar.reset(730)
    action = np.random.default_rng(98).uniform(-.15, .15, scalar.action_dim)
    scalar.step(action)
    start = time.perf_counter()
    for _ in range(steps):
        scalar.step(action)
    scalar_elapsed = time.perf_counter()-start
    batch = scalar.vectorize(num_envs)
    try:
        batch.reset(np.arange(num_envs)+730)
        actions = np.broadcast_to(action, (num_envs, scalar.action_dim)).copy()
        # Compilation and initial thread creation are reported separately.
        start = time.perf_counter()
        batch.step(actions)
        warmup = time.perf_counter()-start
        start = time.perf_counter()
        for _ in range(steps):
            _, _, _, info = batch.step(actions)
        elapsed = time.perf_counter()-start
        warnings = int(np.sum(info['physics_warning_count']))
        scalar_warnings = scalar.measurement()['physics_warning_count']
        if warnings or scalar_warnings:
            raise RuntimeError('Physics warnings invalidate this throughput measurement')
        return {'schema':'xmimic.urr-physics-throughput/v1', 'environment':scalar.binding,
                'num_envs':num_envs, 'workers':batch.workers, 'control_steps':steps,
                'scalar_seconds':scalar_elapsed, 'batch_seconds':elapsed,
                'batch_warmup_seconds':warmup, 'scalar_transitions_per_second':steps/scalar_elapsed,
                'batch_transitions_per_second':steps*num_envs/elapsed,
                'speedup':num_envs*scalar_elapsed/elapsed, 'physics_warning_count':warnings,
                'scope':'CPU simulation collection throughput; excludes learner updates/checkpoint writes; no learning gain claim'}
    finally:
        batch.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--steps', type=int, default=128)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1 or args.output.exists():
        parser.error('positive dimensions and a new report path are required')
    report = benchmark(args.prepared, args.num_envs, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key:value for key,value in report.items() if key != 'environment'}, indent=2))


if __name__ == '__main__':
    main()
