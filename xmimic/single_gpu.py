"""The 1GPU layout: everything on one GPU, ``log_every`` (50) iterations per compiled XLA program.

Every buffer is donated and nothing returns to the host inside a program; the LR-decay trigger is evaluated on the device.
The host dispatches program k+1 before it logs program k, and checkpoints are written from a background thread, so the
GPU never waits for Python. Compilation happens before the training clock starts.
Explicit managed recovery uses synchronous chunk boundaries and complete state snapshots.
"""

from __future__ import annotations

import dataclasses
import os
import time
import signal
from pathlib import Path

import jax

from . import trainer_class
from .config import Config
from .checkpoint import RecoveryOptions, restore_checkpoint, run_binding, save_checkpoint
from .runtime import RunLog, compile_all, freeze_gc, run_pipelined, run_synchronized, write_json


def run(cfg: Config, *, recovery: RecoveryOptions | None = None):
    if recovery is not None:
        if cfg.layout != "1GPU" or cfg.log_every <= 0 or cfg.ckpt_every <= 0:
            raise ValueError("managed recovery requires 1GPU and positive chunk/checkpoint sizes")
        if cfg.iterations % cfg.log_every or cfg.ckpt_every % cfg.log_every:
            raise ValueError("managed iteration/checkpoint boundaries must align with log_every")
    os.makedirs(cfg.log_dir, exist_ok=True)
    meta = {**dataclasses.asdict(cfg), "gpus": cfg.gpus}
    write_json(os.path.join(cfg.log_dir, "args.json"), meta)
    t_proc = time.time()
    xm = trainer_class(cfg)(cfg)
    consts, carry = xm.init(jax.random.PRNGKey(cfg.seed))
    binding = run_binding(cfg, recovery.source_binding) if recovery is not None else None
    start_it, elapsed = 0, 0.0
    if recovery is not None and recovery.resume_from is not None:
        consts, carry, start_it, elapsed = restore_checkpoint(
            Path(recovery.resume_from), consts, carry, binding=binding)
        if start_it > cfg.iterations or start_it % cfg.log_every:
            raise ValueError("checkpoint is not a valid program boundary")
        if recovery.on_restored is not None:
            recovery.on_restored()
    jax.block_until_ready(carry)
    print(f"setup {time.time() - t_proc:.1f} s | motion {xm.motion.num_frames} frames, {xm.motion.num_bins} bins | "
          f"goal {xm.goal_dim} critic_in {xm.critic_in} wm_in {xm.wm_in} | devices {jax.devices()}", flush=True)
    # compile both programs, side by side, before the training clock starts (from the persistent cache: a load)
    t_c = time.time()
    warm, chunk = xm.make_warmup(cfg.warmup_steps), xm.make_chunk(cfg.log_every)
    carry_aval = jax.eval_shape(warm, consts, carry)[0]
    lowered = (warm.lower(consts, carry), chunk.lower(consts, carry_aval))
    warm, chunk = compile_all([lambda lw=lw: lw.compile() for lw in lowered])
    compile_s = time.time() - t_c
    print(f"aot compile: {compile_s:.1f} s (excluded from the training clock)", flush=True)
    write_json(os.path.join(cfg.log_dir, "compile.json"), {"compile_s": compile_s})

    freeze_gc()
    t0 = time.time() - elapsed  # accumulated training clock; compilation remains excluded
    if recovery is None or recovery.resume_from is None:
        carry, _ = warm(consts, carry)
        jax.block_until_ready(carry)
        print(f"warm-up: {cfg.warmup_steps} collection steps {time.time() - t0:.1f} s", flush=True)
    else:
        print(f"restored complete training state at iteration {start_it}", flush=True)
    log = RunLog(xm, cfg, t0, carry["avg"], meta)
    log.it_log = start_it
    state = {"carry": carry, "it": start_it}
    del carry

    def dispatch():
        state["carry"], st, avg_flat = chunk(consts, state["carry"])
        state["it"] += cfg.log_every
        return state["it"], st, avg_flat

    if recovery is None:
        it = run_pipelined(dispatch, log, cfg.iterations, cfg.max_minutes)
    else:
        stopped = {"value": False}
        last_snapshot = {"iteration": None}

        def request_stop(signum, frame):
            stopped["value"] = True

        def should_stop():
            return (stopped["value"] or (recovery.stop_requested is not None and recovery.stop_requested())
                    or (recovery.stop_file is not None and Path(recovery.stop_file).exists())
                    or (cfg.max_minutes > 0 and time.time() - t0 >= cfg.max_minutes * 60))

        def snapshot(iteration):
            if last_snapshot["iteration"] == iteration:
                return
            destination = Path(recovery.directory) / f"state_{iteration:09d}"
            if recovery.resume_from is not None and destination.resolve() == Path(recovery.resume_from).resolve():
                return  # an already admitted boundary is immutable
            save_checkpoint(destination, consts, state["carry"], binding=binding,
                            iteration=iteration, elapsed_seconds=time.time() - t0)
            if recovery.on_checkpoint is not None:
                recovery.on_checkpoint(destination, iteration)
            last_snapshot["iteration"] = iteration

        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)} if recovery.handle_signals else {}
        try:
            for sig in previous:
                signal.signal(sig, request_stop)
            if recovery.resume_from is None:
                snapshot(0)  # retain warmup/replay/randomized model before the first update
            it = run_synchronized(dispatch, log, cfg.iterations, start_iteration=start_it,
                                  max_minutes=cfg.max_minutes, checkpoint_every=cfg.ckpt_every,
                                  checkpoint=snapshot, stop_requested=should_stop)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    print(f"done: {it} iterations in {(time.time() - t0) / 60:.2f} min", flush=True)
