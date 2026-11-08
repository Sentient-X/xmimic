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

The [managed training workflow](managed-training.md) now implements CPU retargeting,
immutable local/Catalog cache custody and single-GPU authored execution. Runtime
publication and live GPU qualification remain separate; this is not a qualified
whole-corpus trainer or physical robot.

## Pinned source model

| Source | Revision | Evidence and limit |
|---|---|---|
| Sentient-X/xmimic | `571715f02c51654dca07b5dfa6651d0e586ca93e` | MIT code and retained third-party notices; original trainer is G1-specific |
| bones-studio/seed | `2f59b2077b9da34dd4e43618e705c7cb962c9a66` | Human source mirrored privately to R2; full source hashes, Catalog sample restore and HF-free inventory passed; [custody evidence](bones-seed.md) |
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

Robot bundles must be derived by the owning embodiment/simulation layer. The nominal exporter now lives in
`sx_embodiments.nominal_mujoco`, with a separately hashed assumption profile. Supplying a string `embodiment_id`
does not authenticate a bundle or prove it was derived from that body. The loader
checks local model consistency, not a platform admission signature.

Current simulator limits remain explicit: one floating root, scalar revolute
coordinates with direct position actuators, 5 ms physics and 50 Hz control.
Prismatic, ball, coupled or passive dynamic coordinates need a supported model
projection before admission. The owner explicitly approved labelled nominal actuator assumptions for simulation;
those assumptions must never overwrite physical drive facts.
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

The owner confirmed there is no newer full-body CAD to wait for. Carry shared
upper-body corrections into the existing full-body assembly: retain its pelvis,
waist roll/pitch, chest and legs, and mate the corrected arm/gripper modules at
its existing shoulder seats. Keep the static body's table clamp, torso drive and
unit calibration out of that transfer. Shared modules own the reusable geometry,
joint limits and drive facts so subsequent corrections propagate to both bodies.

The two descriptions use different chest frames and shoulder widths. A nominal
head placement can be derived by aligning their shoulder centers while preserving
their declared forward/up axes. This is an explicit design choice, not a measured
mount or a claim that the chest shapes coincide. Retaining the old shoulder seats
leaves a roughly 7.369 mm narrower span than the static body. Record this difference
and test the transform rather than silently treating the chest frames as equal.
The corrected full body remains a development assembly pending mounting, collision
and dynamics qualification. Missing new CAD does not prevent composing or testing it.
The source audit of this candidate reports 39 coordinates, seven joints per arm,
and identical part facts for both arms, both grippers, the neck, both head cameras
and the head IMU. Only the torso differs among shared component names, deliberately:
the full body retains its waist chain and original torso drive. The shared arms'
part-local ready references are not an accepted standing or whole-body Ready pose.

The old full-body MJCF has zero actuators, a misspelled hip joint and limits
inconsistent with the canonical body. Preserve it for historical provenance;
it must not describe the corrected assembly or become its training model merely
because it loads. The candidate has no active MJCF; a new simulation projection is
still required.

Reproduce the source audit in an environment containing the owning packages:

```bash
python tools/audit_sentient_embodiment.py
```

This reads verified embodiment assets and compiles the existing MJCF on CPU. It
does not open hardware or return a training-readiness receipt.

## Required next implementation

1. Verify the composed full-body candidate against its source parts and explicitly
   derived mounting assumptions. Prove shared component facts, unchanged lower-body
   FK and motion of the corrected upper parts with the waist; preserve historical
   identities and coordinate order. Generate simulation assets through the owning
   layer, then qualify contacts, collisions, masses and drive parameters.
2. Derive governed training selections from the mirrored SOMA source and temporal
   labels. For each original and mirror, retain source digests, annotations,
   timebase, actor/take/group, license and an explicit disposition. Retarget SOMA
   source against each intended embodiment, checking entire trajectories,
   contacts, limits and feasibility. Keep rejected motions in the accounting.
3. Replace whole-corpus GPU residency with bounded reference shards, coverage-aware
   sampling, persistent per-source failure statistics and batched evaluation.
   Current `load_motion` still loads all selected frames. Full BONES-SEED training
   is not implemented by the metadata inventory or NPZ parser.
4. Qualify the implemented complete restart state on actual GPU/Warp execution.
   CPU/JAX continuation tests preserve optimizers, normalizers, RNGs, replay,
   simulator/randomized parameters, counters, sampler and probes; policy-export
   NPZ checkpoints alone still cannot resume paid training.
5. Bind the program to the platform's authored-trainer contract. Local inspected
   `sx_training.TrainingContext` provides staged data, logging, checkpoint/restore,
   stop handling and artifact output; that implementation is work in progress,
   not verified published/deployed support. The companion owning changes admit CPU transforms and one-GPU learning over
   private named motion artifacts. Multi-GPU authored placement remains unsupported. Do not bypass those restrictions
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
robot motion or deployment has run. Source custody is complete, with an HF-free
Catalog sample restore and complete metadata inventory. CPU retargeting and exact CPU/JAX recovery have local evidence; bounded
GPU training and recovery still require hosted qualification.

Local validation on 2026-11-07: 41 xmimic CPU tests passed, including fast/reference
reward equivalence for unequal contact heights. The companion embodiment candidate
passed 53 focused composition/history/generated-output tests; an independent
refutation corrected stale successor expectations. Neither result is a traversal
through a deployed platform or evidence of physical acceptance.

Following the owner's clarification that no newer full-body CAD exists, the
corrected full-body candidate passed 262 focused owning-package tests. The actual
audit resolves development body
`a2ad7e6e3f2b0cf5a73f1b1255f2e022206a057481ba3b8cb384066b682834b3`.
The companion remains draft and unpinned; it was tested against the checkout's
embodiment baseline, not a reconciled newer `main` or deployed station.

## Shared URR source and physical-site retargeting

The SOMA path now converts complete source observations to the existing
`sx_actions.urr.Motion`; it does not define another motion schema. The explicit
source correspondence initially preserves SOMA joint names as roles. Positions,
rotations, literal-period-derived nanosecond timestamps and the chosen world frame
remain separate from body proportions and joint fitting. Each retained cache includes
`source-motion.json` and its shared Motion identity, including rejected candidates.
Changing robot proportions preserves these source bytes. Requalification carries them
forward without modifying the original cache.

`MotionBinding` resolves source roles to sites owned by the selected embodiment.
The Sentient example uses chest, shoulder, elbow, wrist, hip, knee, ankle and toe sites;
it fits eight limb directions, two ankle-to-toe directions and chest/hand orientation.
Physical sites compose
with the shared modules. A new body still needs an explicit semantic correspondence,
frame calibration and supported simulation/control projection. This is not automatic
anatomy discovery or a claim that every legged embodiment is qualified.

Owner commit `ec1e7fd8248af8ce6ce54ffb2a7ddbc29d9c4ced` adds eight pivot sites, giving
humanoid identity `eb249c00dcff12dc212db1e6122df0ea4f13d57f17fc90d4c4a8855051e036f5`.
Its URDF is byte-identical to the approved `ca3fb7` assembly; removing only the new sites
restores that old identity. This is a new semantic revision, not a new hardware approval.
The upper-body shoulder sites belong to its shared arm modules, so the static body and
humanoid inherit the same facts.

The owner rejected the earlier endpoint-only retarget visually on 2026-11-08. Its
small endpoint residuals missed incorrect limb bends; downstream moving joints were
also used as fixed anchors, shortening modeled upper arms and thighs. That historical
cache is retained as a rejected visual diagnostic and remains unadmitted for training.
A narrow numerical acceptance result is not evidence of faithful motion.

Direction errors now participate in optimization, convergence, frame-journal recovery,
nominal quality and visualization. Journal v2 retains those errors; explicit v1 reading
supports only old zero-direction executions. Nominal projection cannot erase a failed
direction objective. First-pose calibration fits the declared physical point-pair
directions before assigning hand/chest frame offsets. Feet have no fitted orientation
offset: that would absorb a wrong initial foot heading into the target itself.
This nominal calibration does not establish sole contact, no-slip, collision clearance,
balance or a trained policy. XMimic still needs derived joint references for its current
tracker; the unchanged URR source remains available for a future direct task-space
controller consumer.

The historical eight-direction, fitted-foot-orientation599-frame run
passed geometric goals on7frames, with maximum limb-direction63.34degrees and
orientation52.90degrees. It failed direction, orientation and speed checks and retained
only diagnostic output plus the unchanged URR source. Its fitted foot offsets also
hid incorrect absolute heading, so small foot orientation residuals were misleading.
No GPU training was started.

Owner revision `aa3c97f421a961ff4ede1cb5518ece7e25efba78` adds two fixed toe sites
at the midpoints of the literal front sole edges, producing body
`1fb0d53b81fca5fcd8c09b00e796a31e08313fa2ea8bc22b4248dcb8b29eda3c`.
The approved URDF remains unchanged; removing those two sites restores `eb249c`.
The example maps observed `Foot→ToeEnd` to physical ankle→toe-edge directions
using the existing generic `Direction` constraint. These landmarks belong to the
shared leg modules. No joint-name conditional, per-foot rotation, new toe joint or
fitted foot offset is used by the solver. An ankle-to-toe line slopes down and must
not be treated as a horizontal sole axis; one direction also cannot determine full
sole orientation or contact.

The viewer independently measures every source frame with MuJoCo FK. Its
`raw_direction_metrics` compare actual physical point-pair directions to unchanged
observed source directions, including both 3D angle and gravity-plane heading.
These measurements use no fitted orientation offsets, and remain full-frame even
when the display samples a lower frame rate. Undefined projected headings are null.

The same-metric retrospective audit of the old599-frame trajectory found mean
ankle-to-toe angular errors98.62degrees left and92.12degrees right. On frame zero,
errors107.36/94.05degrees become1.54/1.66degrees in the new calibration. This is
only a single-frame improvement: the new calibration remains rejected, with
leg-direction errors9.56–11.30degrees. A separate bounded least-squares diagnostic
from two starts also retained approximately10–11degree leg errors while both
terminal ankle coordinates reached their25degree limits. These local fits do not
prove global infeasibility or justify changing canonical limits. No new full-clip
cache or training admission was produced.

A generic numerical correction prevents false convergence when clipping an
unconstrained least-squares step at a joint limit destroys descent. A projected
gradient fallback can progress while preserving exact bounds and holds. A separate
reachable planar fixture reproduces the old stall and verifies the correction;
it does not establish that the real humanoid target is feasible.

### Generality is an acceptance condition

Embodiment-specific behavior in the retargeting algorithm indicates a missing owner
capability; putting that behavior into an example or configuration file does not fix it.
Bodies supply canonical geometry, native ordering, limits, sites and physical relations.
Source adapters supply observed roles, units, clocks and coordinate frames. Experiments
supply explicit reference configurations, role/site correspondences, root placement,
weights and nominal simulation assumptions. The algorithm must not infer those facts
from a robot name, a joint-name substring or a fixed number of limbs.

Shared actions revision `7507a26c009f3ec10ad25bc10f63e3c7cfa3099d` adds
`MotionBinding.joint_names` (canonical-order site dependencies, including mimic leaders)
and `correspondence_scale` over explicitly declared role chains. The example now holds
only the complement of those dependencies at supplied reference values. Named knee/hip
seed offsets and wrist/ankle terminal-roll holds are removed. A zero Jacobian at a
singular pose does not freeze a coordinate. Support calculations reuse the existing
geometry extent operation, without a Sentient mesh-only branch.

The geometric solver accepts one actual observation for calibration. Training duration
checks remain at the workflow/export boundary, so a static calibration cannot become a
training motion. Tests cover a fixed-base mechanism and three branches, an unobserved
branch, and complete renaming/reordering of joints, links, sites and source roles. These
establish numerical invariance, not support for every articulation or physical dynamics.

Remaining owning-layer gap: `sx_actions.whole_body` combines geometric fitting with
executable action admission, while XMimic maintains an offline geometric implementation.
Consolidation must separate the shared geometric problem/result from action admission,
then make both consumers use it. Remove the duplicated geometric implementations when
both consumers are served; keep frame journals and clip sequencing in XMimic. Do not
fabricate actuator contracts, add a nominal bypass to command admission, or merely move
the offline solver into another wrapper. Until that work and whole-motion qualification
are complete, this fork is not qualified as a general embodiment training capability.
