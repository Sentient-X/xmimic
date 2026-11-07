# Legged embodiment training through SentientX

## Objective and ownership

Train XMimic's SuperMimic-style world-model tracker against registered legged
embodiments, starting with the Sentient humanoid. Retain LAFAN1 as the published
G1 baseline and account for the entire BONES-SEED corpus, including mirrors and
annotations. A human motion may need a different task/retargeting contract on a
quadruped than on a humanoid; equal tensor widths do not establish compatibility.

XMimic owns the algorithm, numerical motion projection and complete learning
state. `sx-embodiments` owns robot geometry, joint order, limits, drive facts and
composition. Catalog owns source artifacts, provenance, access, split membership
and held-out quarantine. The training platform owns job authority, placement,
cost/deadline limits and durable artifacts. Do not copy robot constants here or
create a second scheduler, catalog or checkpoint authority.

This is an implementation in progress. CPU boundary tests and a complete metadata
inventory do not establish a working full-corpus trainer or a qualified robot.

## Pinned source model

| Source | Revision | Evidence and limit |
|---|---|---|
| Sentient-X/xmimic | `571715f02c51654dca07b5dfa6651d0e586ca93e` | MIT code and retained third-party notices; original trainer is G1-specific |
| bones-studio/seed | `2f59b2077b9da34dd4e43618e705c7cb962c9a66` | Authorized v004 metadata inspected; [inventory evidence](bones-seed.md); motion archives not yet imported |
| sx-embodiments inspected baseline | `982813f75255a0bca2c27dbf687c252825fcdebb` | Full-body and corrected static-body definitions diverge; owning composition work is separate |
| Published G1 benchmark | upstream README and `results/` | Reported H100 performance, not reproduced in this change |

The source loop collects simulated transitions, fits a dynamics model, improves
the policy through imagined rollouts, fits a critic and updates reset weights.
Essential semantics are the physical state, action interpretation, reference
timebase, differentiable task, optimization and recovery state. G1 names, exactly
two contact bodies and implicit 29-column joint ordering are incidental.

## Implemented boundaries

- `Config.robot_dir` selects an exported `scene.xml`/`meta.json` robot bundle in
  training, evaluation and CPU rollout. The original G1 bundle remains the default.
- Contact bodies and contact heights are explicit model metadata; any nonempty
  number of tracked contacts is supported at this boundary.
- Mirror augmentation uses an explicit signed permutation, checked to be an
  involution. A body with no verified sagittal symmetry uses `--wm_mirror 0`.
  The G1 mapping preserves the original algorithm's mapping exactly.
- Fixed frames in the differentiable kinematic tree remain fixed rather than
  accidentally indexing the last joint. CPU tests compare four contact frames
  against actual MuJoCo forward kinematics.
- [Named retargeted motion](motion-inputs.md) binds every clip to an exact
  embodiment revision and reorders named coordinates. LAFAN CSV input is refused
  for a nonbaseline robot bundle. Evaluation and rollout refuse a checkpoint's
  declared embodiment mismatch.
- [BONES-SEED inventory](bones-seed.md) preserves every metadata row and groups
  originals/mirrors without dropping motions whose durations differ.

Robot bundles must be derived by the owning embodiment/simulation layer. This
change does not yet implement that exporter. Supplying a string `embodiment_id`
does not authenticate a bundle or prove it was derived from that body. The loader
checks local model consistency, not a platform admission signature.

Current simulator limits remain explicit: one floating root, scalar revolute
coordinates with direct position actuators, 5 ms physics and 50 Hz control.
Prismatic, ball, coupled or passive dynamic coordinates need a supported model
projection before admission. Do not invent actuator gains to pass this boundary.
`meta.json` also carries the existing kinematics, randomization and solver facts
used by XMimic; a correct joint-address check alone does not qualify those facts.

## Sentient composition prerequisite

The old full humanoid and old upper-body variant share an obsolete module with
five-coordinate arms. The corrected static upper body has seven-coordinate arms,
different drive facts and corrected head/gripper geometry. It was an independent
monolithic module, so corrections did not propagate to the full humanoid.

The owning change extracts actual corrected arm, gripper and head modules through
`Subtree`, fitted `Socket`s and `Mate`s. Nested ownership changes native coordinate
order; the preceding document must remain readable under its old identity, never
be silently mapped onto the new order. Generated bindings must follow the new
revision together. This does not authorize a physical rollout.

The companion source candidate is not ready for a superproject pin advance:
the currently inspected station-report resolver requires a compatible successor
even to interpret an old running body. Historical reports must remain readable
without approving new positional commands, or the rollout must explicitly migrate
assignments. Retaining old documents alone does not fix that live consumer.

The remaining source gap is the corrected upper-body attachment to the full
pelvis/waist and whether both waist coordinates remain. The corrected source
contains no legs and uses different chest/base frames. The old full-body MJCF
has zero actuators, a misspelled hip joint and limits inconsistent with the
canonical body. It must not be reused as a training model merely because it loads.

Reproduce the source audit in an environment containing the owning packages:

```bash
python tools/audit_sentient_embodiment.py
```

This reads verified embodiment assets and compiles the existing MJCF on CPU. It
does not open hardware or return a training-readiness receipt.

## Required next implementation

1. Complete the shared full-body assembly from a sourced mount/waist definition.
   Prove shared component facts and FK; preserve historical identities and
   coordinate order. Generate simulation assets through the owning layer.
2. Import pinned BONES-SEED source archives and temporal labels into governed
   storage. For each original and mirror, retain source digests, annotations,
   timebase, actor/take/group, license and an explicit disposition. Retarget SOMA
   source against each intended embodiment, checking entire trajectories,
   contacts, limits and feasibility. Keep rejected motions in the accounting.
3. Replace whole-corpus GPU residency with bounded reference shards, coverage-aware
   sampling, persistent per-source failure statistics and batched evaluation.
   Current `load_motion` still loads all selected frames. Full BONES-SEED training
   is not implemented by the metadata inventory or NPZ parser.
4. Add complete restart state: model/critic/policy and optimizers, normalizers,
   RNGs, replay, simulator and randomized parameters, counters, sampler and probes.
   Existing policy-export NPZ checkpoints cannot resume paid training exactly.
5. Bind the program to the platform's authored-trainer contract. Local inspected
   `sx_training.TrainingContext` provides staged data, logging, checkpoint/restore,
   stop handling and artifact output; that implementation is work in progress,
   not verified published/deployed support. Current local admission supports one
   GPU and native episode corpora; arbitrary motion artifacts and multi-GPU
   placement require owning platform changes. Do not bypass those restrictions
   with direct database writes or a standalone cloud launcher.

## Acceptance and evidence

The first end-to-end acceptance is a governed, bounded one-GPU run for one
registered body and a small admitted split: reproducible retarget → simulation
tracking → interruption → exact recovery → checkpoint evaluation → durable
artifact and cost receipt. Then scale source coverage and placement.

Keep train/probe data separate from untouched evaluation data. Original/mirror
families belong in one partition; use actor/take separation when testing that
generalization. Probe failures influence training, so probe clips are not a held-out
test set. Report all motion dispositions and category/actor coverage, tracking
errors, falls, foot slip, constraint violations, memory, compile/training time and
cost. Fix thresholds and seeds before experiments. The published G1 return target
is a baseline reproduction criterion, not acceptance for every morphology.

At this stage no H100 benchmark, managed-job traversal, policy training, physical
robot motion or deployment has run. Full source download, retargeting, bounded
training and exact recovery remain open.

Local validation on 2026-11-07: 41 xmimic CPU tests passed, including fast/reference
reward equivalence for unequal contact heights. The companion embodiment candidate
passed 53 focused composition/history/generated-output tests; an independent
refutation corrected stale successor expectations. Neither result is a traversal
through a deployed platform or evidence of physical acceptance.
