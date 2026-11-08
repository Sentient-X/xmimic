# URR motion control: experiments and acceptance

Status: experimental program, agreed direction on 2026-11-08. This document
specifies comparisons and acceptance; it is not an implemented benchmark runner,
a completed experiment, or a qualified controller. No comparative scores exist yet.

## Outcome

Convert UMI trajectories, recorded teleoperation episodes and BONES-SEED human
motion into the existing shared URR Motion representation. Train controllers on
collections of these trajectories, then execute URR targets with measured accuracy
on canonical static bimanual, semi-humanoid and mobile embodiments. Targets include
tool/gripper/hand pose, measured gripper aperture, and other explicitly mapped roles.
The source trajectory remains independent of the target body.

The first generality claim is one interface and training recipe with separately
trained, embodiment-bound controllers. One set of weights for unseen morphologies
requires a separate held-out-embodiment experiment. Unsupported actuation, topology
or task requirements are reported explicitly rather than silently approximated.

## Baselines determine the combination

Preserve both upstream methods as executable controls. Combining their ideas is a
hypothesis, not a guaranteed improvement. A faster run, higher aggregate reward or
better average pose error cannot compensate for a regression in required behavior.

Two comparison tracks are necessary:

| Track | Required controls | What it establishes |
| --- | --- | --- |
| Native reproduction | Unmodified upstream XMimic on its published G1/LAFAN task; unmodified PolicyIK on its released OpenArm task | The environment and evaluation reproduce the respective starting points. Their different tasks do not support a direct ranking. |
| Matched URR tasks | Explicitly labelled XMimic-derived and PolicyIK-derived adaptations, combined recipe, and bounded numerical IK/controller baseline | Differences under the same embodiment, targets, initial states, physical model and evaluation. An adaptation is never labelled vanilla. |

Pin code commits, checkpoints, asset and task digests, runtime image, solver/driver
versions and native evaluation protocol before running. Published results are
reference values, not local measurements. Preserve native single-attempt results
separately from retry/recovery protocols. Mark unsupported native cells N/A; do not
invent a floating joint to make an upstream fixed/mobile comparison appear valid.

In the matched suite, first measure a 2-by-2 experiment: model-free versus learned-
world-model training, with versus without the proposed precision-control package.
Keep URR conditioning and physical action limits common. Then remove command-state
feedback, multiscale errors, near-goal action scaling and progress/holding terms
individually where applicable. Every accumulated-command variant exposes that state
to its policy and dynamics model. The intentionally incomplete-state ablation is
diagnostic only. These experiments identify benefits and harmful interactions;
the final architecture follows their results rather than a predetermined fusion.

Changes needed for a common task must be enumerated: observations, action semantics,
reward, termination, mounting and simulator. Evaluate canonical task-space errors
outside the learner so changed reward scales cannot change the definition of success.
Include delayed feedback, perturbations, moving goals and sustained holds: slowing
near a hand target must not suppress balance corrections or gripper responsiveness.

## Common inputs and execution contract

Reuse `sx_actions.urr.Motion`, `MotionFrame`, Position/Pose and aperture semantics.
Do not introduce a second URR schema in this fork. The current shared
`observed_frame` path already adapts explicitly framed pose observations;
episode-wide extraction, synchronization and training integration still require proof.
The current SOMA adapter retains observed poses and native timestamps. Existing
capability is not evidence that every UMI/teleop recording has a qualified importer.

For every source, retain original artifact identity, timebase, coordinate convention,
frame calibration, selected roles and conversion provenance. Position-only observations
remain position-only; missing orientation, aperture or contact is not zero. Do not
interpolate across unobserved gaps. Record robot-derived FK poses as derived evidence,
including the source body/state identity; never relabel teleoperator commands as
measured achieved poses. Gripper width is in metres, with its physical definition
and owner-provided mapping to native actuation. A hand, wrist, tool centre and jaw
centre are distinct frames, connected by explicit transforms. Aperture is not grasp
force or proof that an object was successfully held.

Keep source observations, target correspondence and controller result separate.
Role/site mappings, task tolerances, priorities and feasible timing are explicit,
versioned task inputs. A source that has no feet imposes no fabricated foot target;
the selected body's support/balance requirements still apply. Required unsupported
roles fail preparation rather than silently disappearing. A portable policy artifact
binds body revision, native action layout, controller state, role/frame conventions,
training selection and evaluation evidence. Check those at deployment and retain
achieved URR observations and native state/action evidence during execution.

## Datasets and paired evaluation

Start with canonical FK-generated reachable targets and trajectories to separate
controller failures from cross-morphology correspondence failures. Then add real
UMI, teleoperation and BONES-SEED selections without changing their source bytes.
Test pose-only, aperture-only, coordinated pose/aperture, two-hand relative motion,
moving trajectories, holds, and explicitly unreachable or unavailable targets.

Freeze evaluation membership through Catalog before training. Keep all mirrored,
resampled, retargeted and augmented versions of one source family in the same split.
Use validation data for tuning; never select checkpoints or tolerances on final test
results. Separate held-out trajectories on a known body from held-out embodiments.
Retain rejected inputs and failed attempts in coverage accounting; report both
eligible-task performance and end-to-end yield over the original source selection.

Pair initial states, targets, disturbances and evaluation seeds between methods.
Use at least three independent training seeds for screening; choose confirmatory
sample size from a separate pilot's variance and the predeclared regression margin.
Reuse the same tuning budget. Compare both equal compute budgets and time/cost to
the same target, retaining compilation, data preparation, training and evaluation
costs separately, plus total cost. Native reproduction also retains its original
published compute protocol. GPU minutes alone do not establish equivalent work.

## Independent measurements and promotion

Record per role, per task and per embodiment, rather than one blended reward:

| Requirement | Measurements |
| --- | --- |
| Pose precision | Position error in mm; geodesic orientation error in degrees; median, p95 and maximum; simultaneous tolerance success |
| Gripper control | Aperture error in mm, response lag, overshoot, sustained accuracy and pose accuracy while opening/closing |
| Temporal tracking | Timestamp-aligned error, phase/response lag, recovery time, smoothness and source-time versus deliberately retimed results |
| Holding | Continuous dwell inside every required tolerance; drift and jitter; single attempt versus retries |
| Mobility and contacts | Falls, support/contact failures, slip, penetration and task-relative base/foot motion |
| Constraints | Joint/velocity/effort violations, collisions, saturation and infeasible-target behavior |
| Operations | Inference latency distribution, throughput, GPU-hours, wall time, cost and recovery/checkpoint completeness |

Before each run, select primary metrics, absolute task tolerances, practical
non-inferiority margins, confidence level, seed/sample counts and stopping criteria.
For a lower-is-better metric, require the upper confidence bound on the paired
candidate-minus-baseline difference to be within its allowed margin; reverse the
direction for higher-is-better metrics. Account for training-seed and trajectory
grouping, and predeclare handling of multiple primary comparisons. Missing evidence
or a wide interval means inconclusive, not no regression. Report all co-primary
metrics separately; a locomotion gain cannot hide a gripper regression. Also require
absolute performance and constraint criteria, since two poor methods can be equal.

No promoted combination may degrade a required capability beyond its predeclared
margin relative to the relevant validated control. Measured tradeoffs are reported
for an explicit product decision; do not change margins after seeing a result.
Finite benchmarks cannot guarantee the absence of every possible regression.

Simulator pose accuracy, encoder-FK accuracy and externally measured physical
accuracy are separate results. Physical precision needs calibrated independent
measurement and authorized hardware trials. Neither imported trajectories nor
simulated policy success establishes successful manipulation or physical deployment.

## Resolve the current humanoid residual without hiding it

The retained first-pose calibration has about 1.5–1.7 degree ankle-to-toe error but
about 10–11 degree leg-direction error. Those are geometric correspondence errors,
not measured closed-loop tracking errors. Original/current knee axes agree; there
is no demonstrated sideways-knee defect. Fixed pivot offsets make anatomical
segment directions and native joint angles different quantities. Both terminal
ankle-roll coordinates reach their limits in the independent local fit. Two local
solutions do not prove global infeasibility.

Next distinguish three causes with separate evidence:

1. Use FK-generated feasible trajectories to measure solver/controller performance
   without a human-to-robot correspondence problem.
2. Compare source anatomical roles to owner-defined functional sites and frames.
   Review segment-direction objectives separately from tool and support objectives.
   A single toe direction does not constrain full sole orientation.
3. Evaluate actual foot placement/orientation, contact/slip, balance, collision,
   joint limits and downstream tracking over complete diagnostic clips. Visual
   improvement alone does not qualify the reference.

If exact anatomical imitation is unnecessary for a defined manipulation task,
declare posture/style as a secondary objective before its experiment, while keeping
required tool/contact constraints explicit. That produces a new task contract and
new evaluation; it does not retroactively accept the old failed cache. Do not widen
the existing 5-degree gate, change native limits, relabel source observations or
teach the learner to reproduce an unreviewed mapping. The current cache remains
unadmitted. Known-feasible synthetic and measured robot trajectories can establish
controller baselines while this source correspondence is investigated.

## Delivery order and present limits

1. Freeze executable native baselines and an independent evaluator; reproduce them.
2. Prove recorded-source-to-URR conversion, including actual aperture channels,
   frame transforms, missingness and timestamp alignment.
3. Refactor the current floating-root/joint-reference-only training boundary for
   declared mounting/mobility and direct URR objectives. Preserve native baseline paths.
4. Run matched static bimanual pose/aperture experiments and component ablations.
5. Repeat through the same interfaces on semi-humanoid/mobile bodies, adding their
   declared dynamics/contact requirements; then qualify precise physical execution.

Use existing SentientX Jobs, Modal placement, persistent checkpoints, local cache
and Catalog/R2 lineage. No separate cloud launcher or parallel robot registry.
Every run has immutable inputs, deadline and spend cap; a model run that may cost
more than $30 needs the owner's confirmation before dispatch. No paid experiment
or physical movement is authorized merely by this plan's existence.

References: [vanilla XMimic](https://github.com/nico-bohlinger/xmimic),
[PolicyIK code](https://github.com/Tinaliu0123/PolicyIK),
[PolicyIK paper](https://arxiv.org/html/2611.05372v1),
[SuperMimic](https://supermimic.github.io/),
[managed execution](managed-training.md), [current motion input](motion-inputs.md).
