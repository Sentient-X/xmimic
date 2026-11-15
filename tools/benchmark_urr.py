#!/usr/bin/env python3
"""Run immutable stationary URR physics controls and labelled learning adaptations.

Native upstream reproductions use tools/benchmark_native.py. This command cannot
relabel adaptations as vanilla, or promote a combination without native evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xmimic.artifact_cache import _atomic, _json, content_hash
from xmimic.urr_benchmark import TaskConfig, URRPhysicsEnv, digest, evaluate
from xmimic.urr_learning import LearningConfig, load_policy, train

VARIANTS = {"dls": (None, False, True), "ppo_coarse": ("ppo", False, False),
            "ppo": ("ppo", True, True), "world_model": ("world_model", False, False),
            "combined": ("world_model", True, True)}


def run(prepared: Path, output: Path, *, methods: list[str], seeds: list[int], evaluation_seeds: list[int],
        learning: LearningConfig, task: TaskConfig, native_results: list[Path] = (), stop_requested=None):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if "raw" in output.parts or output.is_symlink() or prepared == output:
        raise ValueError("Benchmark needs a separate regular derived output directory")
    if not methods or len(set(methods)) != len(methods) or any(m not in VARIANTS for m in methods):
        raise ValueError("Select distinct implemented matched controls; native baselines use their own runtime")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Select distinct non-negative training seeds")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request = {"schema": "xmimic.urr-benchmark-run/v1", "prepared": str(prepared),
                   "preparation_sha256": digest(prepared / "benchmark.json"), "body_sha256": digest(prepared / "embodiment.json"),
                   "methods": methods, "seeds": seeds, "evaluation_seeds": evaluation_seeds,
                   "learning": asdict(learning), "task": asdict(task),
                   "runner_sha256": digest(Path(__file__)),
                   "environment_sha256": digest(Path(sys.modules[URRPhysicsEnv.__module__].__file__)),
                   "learner_sha256": digest(Path(sys.modules[train.__module__].__file__)),
                   "native_receipts": {str(p.resolve()): digest(p) for p in native_results}}
        # JSON round-trip preserves tuples/list equality when reopening a run.
        request = json.loads(json.dumps(request))
        request_file = output / "request.json"
        if request_file.exists() and json.loads(request_file.read_bytes()) != request:
            raise ValueError("Run directory belongs to different code, inputs, methods or settings")
        _atomic(request_file, _json(request))
        records = []
        began = time.monotonic()
        from dataclasses import replace
        for method in methods:
            backend, precision, accumulate = VARIANTS[method]
            for seed in (seeds if backend else seeds[:1]):
                directory = output / method / f"seed-{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                configuration = replace(task, precision=precision, accumulate=accumulate, partition="train")
                def factory(c=configuration):
                    return URRPhysicsEnv(prepared, c)
                receipt_file = directory / "training.json"
                if backend:
                    checkpoints = sorted(directory.glob("checkpoint-*/manifest.json"))
                    resume = checkpoints[-1].parent if checkpoints else None
                    result = train(factory, method=backend, precision=precision,
                                   config=replace(learning, seed=seed), output=directory, resume_from=resume,
                                   stop_requested=stop_requested)
                    _atomic(receipt_file, _json(result.receipt))
                    if not result.receipt["completed"]:
                        _atomic(output / "interrupted.json", _json({"method": method, "seed": seed, "checkpoint": str(result.checkpoint)}))
                        return {"complete": False, "checkpoint": str(result.checkpoint)}
                    saved_binding = json.loads((result.checkpoint / "manifest.json").read_bytes())["binding"]
                    if saved_binding["environment"] != factory().binding:
                        raise ValueError("Checkpoint belongs to a different body, task or controller contract")
                    policy, identity = load_policy(result.checkpoint), digest(result.checkpoint / "manifest.json")
                else:
                    policy, identity = None, None
                def evaluation_factory(c=configuration):
                    return URRPhysicsEnv(prepared, replace(c, partition="evaluation"))
                measured = evaluate(evaluation_factory, policy, evaluation_seeds, directory / "evaluation", policy_identity=identity)
                record = {"method": method, "training_seed": seed, "evaluation": measured,
                          "training_receipt_sha256": digest(receipt_file) if backend else None}
                records.append(record)
                print(json.dumps({"method": method, "seed": seed, "held": measured["held_tasks"], "tasks": measured["tasks"],
                                  "terminal_errors": measured["errors_terminal_window"]}), flush=True)
        native = [json.loads(path.read_bytes()) for path in native_results]
        result = {"schema": "xmimic.urr-benchmark-result/v1", "request_sha256": content_hash(request),
                  "records": records, "native_evidence": native,
                  "complete": True, "elapsed_seconds": time.monotonic() - began,
                  "promotion": {"status": "not_assessed", "reasons": [
                      "This screening runner has no confirmatory paired confidence analysis yet.",
                      "Native reproduction results must independently meet their frozen protocols.",
                      "Absolute task acceptance and every required capability must pass before promotion."]},
                  "scope": "Nominal stationary URR physics experiment; synthetic reachable targets; adapted learners; no physical qualification"}
        _atomic(output / "result.json", _json(result))
        return result


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=tuple(VARIANTS), default=["dls", "ppo", "world_model", "combined"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--episode-steps", type=int, default=100)
    parser.add_argument("--hold-steps", type=int, default=25)
    parser.add_argument("--target-fraction", type=float, default=.08)
    parser.add_argument("--moving", action="store_true")
    parser.add_argument("--native-result", type=Path, action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    result = run(args.prepared, args.output, methods=args.methods, seeds=args.seeds,
                 evaluation_seeds=args.evaluation_seeds,
                 learning=LearningConfig(iterations=args.iterations, rollout_steps=args.rollout_steps, num_envs=args.num_envs),
                 task=TaskConfig(episode_steps=args.episode_steps, hold_steps=args.hold_steps,
                                 target_fraction=args.target_fraction, moving=args.moving), native_results=args.native_result)
    print(json.dumps({"complete": result["complete"], "result": str(args.output / "result.json")}), flush=True)
