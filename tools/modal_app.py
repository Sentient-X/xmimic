"""Train XMimic directly on Modal: one GPU function, one Volume, a cost ledger and resumable state.

    modal run tools/modal_app.py::fetch_lafan                       # G1 + 40 LAFAN1 clips into the volume (CPU)
    modal run tools/modal_app.py::train --name g1-smoke --gpu H100 --args "--iterations 600 --max_minutes 3"
    modal run tools/modal_app.py::train --name sentient-v1 --gpu H100 \\
        --robot sentient-ca3fb7-ds1 --motions bones-batch1 --args "--iterations 3500 --max_minutes 25"
    modal volume get xmimic-runs runs/sentient-v1 ./results/...      # checkpoints, train_log.csv, args.json

Volume ``xmimic-runs`` layout: ``robots/<name>/`` (scene.xml, meta.json, meshes), ``motions/<name>/`` (retargeted npz or
LAFAN1 csv), ``runs/<name>/`` (ckpt_*.npz, train_log.csv, state/state_<it>/), ``cache/jax`` and ``cache/warp`` (persistent
compilation caches, so a second run loads its programs instead of compiling them).

Paid work is kept (sx law 9): ``--max_minutes`` stops training inside the function's ``timeout``; the complete training
state is snapshotted every ``--snapshot_every`` iterations and at the stop (policy checkpoints stay every ``--ckpt_every``), and the next call
with the same ``--name`` and identical arguments resumes from the newest snapshot instead of starting fresh. A call with
different arguments under an existing name is refused (change the name). Every call appends its GPU seconds and list price
to the local ledger ``--ledger`` (default results/demo-2026-11-16/costs.json) and refuses to start past ``--budget``.

The managed sx Job path (docs/managed-training.md) has not completed a GPU run; this app is the direct route until it does.
"""

from __future__ import annotations

import json
import os
import shlex
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOL = modal.Volume.from_name("xmimic-runs", create_if_missing=True)
MNT = "/vol"
# Modal list prices (2026-11): GPU $/s plus the 4 cores and 32 GiB each function reserves.
GPU_PER_S = {"H100": 0.001097, "A100-80GB": 0.000694, "L40S": 0.000542, "A10G": 0.000306}
CPU_PER_S = 4 * 0.0000131 + 32 * 0.00000222
TIMEOUT_S = 45 * 60

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libegl1", "libgl1", "libgomp1", "curl")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .env({"PYTHONUNBUFFERED": "1", "MUJOCO_GL": "egl", "PYTHONPATH": "/opt/xmimic",
          "JAX_COMPILATION_CACHE_DIR": f"{MNT}/cache/jax", "WARP_CACHE_PATH": f"{MNT}/cache/warp",
          "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
          "JAX_INCLUDE_FULL_TRACEBACKS_IN_LOCATIONS": "false"})
    .add_local_dir(str(ROOT / "xmimic"), "/opt/xmimic/xmimic", ignore=["__pycache__"])
    .add_local_file(str(ROOT / "train.py"), "/opt/xmimic/train.py")
    .add_local_dir(str(ROOT / "assets" / "g1"), "/opt/xmimic/assets/g1", ignore=["assets"])
)
app = modal.App("xmimic-train", image=image)


@app.function(volumes={MNT: VOL}, timeout=20 * 60, cpu=2)
def fetch_lafan_remote():
    """The 40 G1 LAFAN1 clips and the G1 meshes from Hugging Face into the volume (never redistributed)."""
    import re
    import subprocess

    hf = "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset"
    api = "https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main/g1"
    clips, meshes = Path(MNT, "motions/lafan1_g1"), Path(MNT, "robots/g1/assets/meshes")
    clips.mkdir(parents=True, exist_ok=True)
    meshes.mkdir(parents=True, exist_ok=True)
    listing = json.loads(subprocess.run(["curl", "-sf", api], check=True, capture_output=True).stdout)
    for item in listing:
        subprocess.run(["curl", "-sfL", "-o", str(clips / Path(item["path"]).name), f"{hf}/resolve/main/{item['path']}"], check=True)
    scene = Path("/opt/xmimic/assets/g1/scene.xml").read_text()
    for mesh in sorted(set(re.findall(r'file="meshes/([^"]*)"', scene))):
        subprocess.run(["curl", "-sfL", "-o", str(meshes / mesh), f"{hf}/resolve/main/robot_description/g1/meshes/{mesh}"], check=True)
    for f in ("scene.xml", "meta.json"):
        Path(MNT, "robots/g1", f).write_bytes(Path("/opt/xmimic/assets/g1", f).read_bytes())
    os.symlink(f"{MNT}/robots/g1/assets", "/opt/xmimic/assets/g1/assets")  # proves the training-time link works
    VOL.commit()
    return {"clips": len(list(clips.glob("*.csv"))), "meshes": len(list(meshes.iterdir()))}


def _train(name: str, argv: list[str], robot: str, motions: str, snapshot_every: int) -> dict:
    import threading

    import jax

    from xmimic import single_gpu
    from xmimic.checkpoint import RecoveryOptions
    from xmimic.config import parse_args

    VOL.reload()
    run = Path(MNT, "runs", name)
    fmt = "lafan1_csv" if robot == "g1" else "retargeted_npz"
    robot_dir = "/opt/xmimic/assets/g1" if robot == "g1" else f"{MNT}/robots/{robot}"
    if robot == "g1":  # the trainer insists on the packaged G1 path; the meshes live on the volume
        os.symlink(f"{MNT}/robots/g1/assets", "/opt/xmimic/assets/g1/assets")
    cfg = parse_args(["--log_dir", str(run), "--robot_dir", robot_dir, "--motion_dir", f"{MNT}/motions/{motions}",
                      "--motion_format", fmt, *argv])
    def program(args):  # --max_minutes is operational (the training clock carries over on resume): a resume may extend it
        return [a for i, a in enumerate(args) if a != "--max_minutes" and (i == 0 or args[i - 1] != "--max_minutes")]

    request = {"argv": program(argv), "robot": robot, "motions": motions}
    run.mkdir(parents=True, exist_ok=True)
    req_file = run / "request.json"
    stored = json.loads(req_file.read_text()) if req_file.exists() else None
    if stored is not None and {**stored, "argv": program(stored["argv"])} != request:
        raise ValueError(f"run {name} exists with different arguments; use a new name")
    req_file.write_text(json.dumps(request, indent=1))
    snaps = sorted((run / "state").glob("state_*")) if (run / "state").exists() else []
    resume = str(snaps[-1]) if snaps else None
    print(f"devices {jax.devices()} | resume from {resume}", flush=True)
    stop = threading.Event()

    def commit_loop():  # checkpoints reach the volume while training runs
        while not stop.wait(90):
            try:
                VOL.commit()
            except Exception as error:  # a failed commit is retried at the next tick and at the end
                print(f"volume commit: {error}", flush=True)

    threading.Thread(target=commit_loop, daemon=True).start()
    t0 = time.time()
    try:
        single_gpu.run(cfg, recovery=RecoveryOptions(directory=str(run / "state"), resume_from=resume,
                                                     snapshot_every=snapshot_every))
    finally:
        stop.set()
        keep = sorted((run / "state").glob("state_*"))[:-1]  # only the newest complete snapshot is kept
        for old in keep:
            for f in old.iterdir():
                f.unlink()
            old.rmdir()
        VOL.commit()
    return {"train_s": time.time() - t0, "ckpts": sorted(p.name for p in run.glob("ckpt_*.npz"))}


GPU_FN = dict(volumes={MNT: VOL}, timeout=TIMEOUT_S, cpu=4, memory=32 * 1024)


@app.function(gpu="H100", **GPU_FN)
def train_h100(*a):
    return _train(*a)


@app.function(gpu="L40S", **GPU_FN)
def train_l40s(*a):
    return _train(*a)


TRAIN = {"H100": train_h100, "L40S": train_l40s}


def _ledger(path: Path) -> list[dict]:
    return json.loads(path.read_text())["calls"] if path.exists() else []


def _book(path: Path, entry: dict) -> float:
    calls = _ledger(path) + [entry]
    total = round(sum(c["usd"] for c in calls), 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"total_usd": total, "calls": calls}, indent=1))
    return total


@app.local_entrypoint()
def fetch_lafan():
    print(fetch_lafan_remote.remote())


@app.local_entrypoint()
def train(name: str, gpu: str = "H100", args: str = "", robot: str = "g1", motions: str = "lafan1_g1",
          ledger: str = "results/demo-2026-11-16/costs.json", budget: float = 7.5, snapshot_every: int = 1000,
          timeout_min: int = TIMEOUT_S // 60):
    path = Path(ledger).expanduser()
    spent = sum(c["usd"] for c in _ledger(path))
    timeout_s = min(timeout_min * 60, TIMEOUT_S)
    worst = (GPU_PER_S[gpu] + CPU_PER_S) * timeout_s
    if spent + worst > budget + 1.0 or spent >= budget:
        raise SystemExit(f"ledger {spent:.2f} USD: launching up to {worst:.2f} more would pass the {budget:.2f} budget")
    t0 = time.time()
    status, out = "failed", None
    try:
        out = TRAIN[gpu].with_options(timeout=timeout_s).remote(name, shlex.split(args), robot, motions, snapshot_every)
        status = "ok"
    finally:  # the wall time of the call bounds its billed time (cold start included): a conservative ledger
        s = time.time() - t0
        usd = round(s * (GPU_PER_S[gpu] + CPU_PER_S), 4)
        total = _book(path, {"name": name, "gpu": gpu, "seconds": round(s, 1), "usd_per_h": round(GPU_PER_S[gpu] * 3600, 3),
                             "usd": usd, "status": status, "args": args, "robot": robot, "motions": motions,
                             "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print(f"{name} on {gpu}: {s:.0f} s, {usd:.2f} USD, ledger {total:.2f} USD | {status} {out}")
