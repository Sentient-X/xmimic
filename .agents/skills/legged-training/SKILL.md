---
name: legged-training
description: Use when adapting XMimic to a SentientX legged embodiment, preparing BONES-SEED, checking robot/motion identity, or continuing the Sentient humanoid integration.
---

# Legged training

Read `docs/sentientx-integration.md`, `docs/motion-inputs.md` and
`docs/bones-seed.md` before extending this path. Robot facts belong to
`sx-embodiments`; XMimic consumes their simulation projection. Never repair a
full-body model by copying corrected upper-body constants into this repository.

## Reproduce the observed boundary checks

Use a Python environment with NumPy, JAX, MuJoCo and pytest. The CPU checks do not
need NVIDIA GPUs or source-motion downloads:

```bash
JAX_PLATFORMS=cpu python -m pytest tests/test_legged_models.py tests/test_motion_inputs.py tests/test_prepare_bones_seed.py -q
```

On 2026-11-07 these passed 41 tests on the development environment (JAX 0.11.0,
MuJoCo 3.11.0). This was not the upstream pinned CUDA/H100 environment. The
kinematics check uses a four-contact MuJoCo model as an independent oracle.

With the actual owning embodiment packages installed:

```bash
python tools/audit_sentient_embodiment.py
```

The initial audit found 25 coordinates in the old full humanoid, five per arm,
versus seven per arm in the corrected static body. Its bundled MJCF compiled
as nq=35, nv=34, nu=0, with a missing correctly spelled right hip-roll coordinate
and 24 limit mismatches. Compilation alone is not usable dynamics.

## Inventory the whole dataset

Precondition: existing HF login with approved `bones-studio/seed` access.
Do not print credentials or store them in manifests.

```bash
uv run --no-project --with huggingface-hub --with pyarrow python tools/prepare_bones_seed.py
```

The real pinned v004 metadata run produced 142,220 motions, 71,132 source groups,
522 actors and zero unresolved mirror parents. Private outputs remain under
ignored `data/bones_seed/inventory/`. This downloads metadata only. Source archive
identity is recorded, not a claim that its bytes have been acquired or verified.

| Failure/signature | Established cause | Required correction |
|---|---|---|
| BONES CSV rejected as nonnumeric/incorrect width | Headered Euler/DOF source is not LAFAN's 30 Hz qpos CSV | Inspect actual source bytes, then explicit source conversion/retargeting |
| Missing `move_g1_mujoco_path` | Published examples lag v004 | Use actual `move_g1_path`; pin metadata bytes |
| Label joins fail using `move_name` | It differs from `filename` in 52,508 rows | Join by `filename`, preserving both |
| Mirrored durations differ | 7,653 real mirror pairs have differing frame counts | Group by validated source family, not equal length |
| Custom body accidentally accepts G1 references | Joint count is not embodiment identity | Bind NPZ to the exact body and named coordinates; refuse CSV for custom models |
| Fast learner fails importing `CONTACT_Z` | Reference reward was generalized but fast reward retained the old constant | Change both kernels and compare rewards at unequal contact heights |
| Later checkpoint is evaluated as the first body | Only the first checkpoint was validated | Check every batched checkpoint's identity and observation architecture |
| Full humanoid misses upper-body corrections | Two source definitions, not one shared corrected module | Use reusable owning-layer modules and a sourced full-body attachment |
| A fitted-module refactor changes joint positions in vectors | Native order follows nested component ownership | Retain exact historical document; regenerate bindings; never alias old identity to new order |

## Stop conditions and evidence limits

No training with unknown attachment geometry, invented drive parameters, mismatched
embodiment identity or unverified retargeting. Every motion remains accounted for
even when infeasible. Preserve held-out splits and original/mirror families.
The current library still materializes all selected frames on the GPU; inventory
completion does not establish full-corpus training scalability. Policy exports
are not recovery checkpoints. Verify the authored trainer API is actually released
before reporting managed integration. No physical motion or run expected to exceed
the repository's approval threshold starts without the user's authorization.
