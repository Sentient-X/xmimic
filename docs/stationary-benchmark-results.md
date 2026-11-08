# Stationary URR controller screening

Status: the benchmark runner executes actual nominal MuJoCo physics. The first
screening is complete. The larger structured-model cohort was interrupted after a
proven reward defect; its partial results and checkpoints remain intact. The new
source-derived reward cohort is complete and remains unpromoted: precision PPO held
15/24 tasks, combined held 12/24. Native vanilla
XMimic and PolicyIK execution is a separate managed-runtime track and has no
completed scores yet. None of these results qualifies physical robot operation.

## Prepared body and measurement

The first fixture composes two canonical PiPER arms through `sx-embodiments`,
using explicit nominal fixed placements at Y=+/-0.35 m. Body identity is
`0c19cd156a9e6010a40cc5d6ba24755351ac43cb6893c785bf80d5a3b097df48`.
Owner revision `83f32c4a5f35b43630230b90402147908f6d9acd` adds fixed-base,
prismatic and declared affine mimic support to the existing nominal exporter.
There are14 controlled coordinates:12 revolute arm joints and2 metric gripper
slides. Two follower slides use the original URDF mimic relations. Tool sites and
aperture laws come from canonical parts. Fixture placements and PD parameters are
simulation assumptions; self/inter-arm collision qualification is excluded.

The Sentient upper body's aperture law remains undocumented. ALOHA was excluded
when its declared native slide bounds/mimic relation disagreed with its URDF.
Neither was repaired by guessing a width law. PiPER's narrower rounded native
bounds are contained in source URDF bounds and are exported unchanged.

Independent nonzero FK matches every link to1.6e-16 m and6.7e-16 rotation-matrix
entries. Both physical jaws follow the declared aperture relation in simulation.
Constant-command PD settles with16.029 mm /3.338 degree hand error from gravity
sag; it does not provide exact pose regulation by itself.

The evaluator records actual native coordinates, achieved URR poses/apertures,
requested URR targets, timestamped errors, joint bounds, forces, velocities and
physics warnings. It refreshes MuJoCo FK after integration; no stale pose is
scored against the next timestamp. Each result is bound to its request, body,
scene, policy and code identities. Learner reward never decides acceptance.

## First completed screening

Source: `ec27776` plus learner revision `22ec946`; native worker changes in later
commits do not change this experiment. Complete historical checkout and file hashes
are retained beside the results. Each learned cell uses20 updates of128 real
transitions:2,560 physics transitions per training seed, far below either upstream
training scale. These are labelled adaptations, not vanilla reproductions.

Eight common synthetic reachable target tasks per cell,250 control steps at50 Hz,
with the final50 steps required continuously inside all tolerances:
position1 mm, orientation0.2 degree, aperture1 mm on both hands. Evaluation seeds
0–7 are development evaluation; they are not final confirmation or sources of
training transitions. Learned controls use training seeds0,1,2; deterministic
numerical IK needs only one control cell. Full-native action authority and a common
per-cycle reference governor avoid restricting only the absolute-action controls.

| Method | Held tasks | Tasks with recorded joint-bound/physics-warning failure |
| --- | --- | --- |
| Bounded numerical IK feedback | 8/8 | 0/8 |
| Coarse PPO adaptation | 0/24 | 0/24 |
| PolicyIK-derived precision PPO adaptation | 0/24 | 19/24 |
| XMimic-derived world-model adaptation | 0/24 | 0/24 |
| Initial combined adaptation | 0/24 | 14/24 |

All13 cells completed in166.78 s on the local CPU, including evaluation and
checkpoint serialization. This is an execution measurement, not upstream training
speed or a converged-learning comparison. The initial combined recipe must not be
promoted. Its failures do not establish that the research combination is impossible.
The successful numerical control demonstrates that these static nominal targets
are physically controllable and that aperture/pose evaluation connects correctly.

Every completed training iteration retains parameters, optimizer state, measured
transition replay, RNG, environment integration state, command state and reward
history. Exact interrupted/continued results are checked against uninterrupted
physics training. Completed evaluation task caches cannot be reused for another
policy or request. Before each algorithm change, retain the old source and results;
never regenerate the first screening with newer code under its old identity.

Private local artifacts:
- `data/stationary-benchmark/screening-v1/result.json`
- `screening-v1/screening-report.json`: per-role terminal metrics and paired
  training-seed/task-family bootstrap intervals, explicitly descriptive screening.
- `screening-v1/checkout/` and `checkout-receipt.json`: complete historical source.
- `screening-v1/visual-review-task-0-v3/`: measured target/DLS/combined replay,
  contact sheet and error curves. The target panel shows requested frames rather
  than an invented target robot configuration.

## Verified private custody of the first screening

The first screening was published through the existing Catalog publisher to private
pack `xmimic.benchmarks.stationary.screening-v1.ec27776bf3b8`, project
`b98ba48a-39c1-4346-98a4-d9c130144a42`. Manifest SHA-256:
`69663086f02ba41de14c2b87e740c33bef7916a8f1dcc77e05454e8151898e47`.
All 8 published objects, 54,948,901 bytes, were downloaded and verified; their archive
inventories cover 1,937 logical files. This is a complete readback of this evidence
pack, not the BONES-SEED dataset or a later benchmark.

The pack retains prepared body/assets, all task and training receipts, each learned
cell's latest complete recovery checkpoint, historical source, owner source/licenses,
and measured visualization v3. Earlier local checkpoints remain untouched. Purpose
is `EVALUATION`, `training_eligible=false`; publication grants no learning admission.
The local receipt is `data/stationary-benchmark/screening-v1-catalog/publication-receipt.json`.

## Interrupted million-transition cohort

`vector-million-v1` preserves source checkout `bcae2a6` and its immutable request.
Each fully trained seed used 500 updates × 64 environments × 32 steps = 1,024,000
real transitions. It used the same development tasks and strict held-pose criteria.
It was stopped after the reward audit below, not completed and relabelled as a new
experiment.

| Method | Completed training/evaluation coverage | Strict held tasks |
| --- | --- | --- |
| Numerical IK | One deterministic control cell | 8/8 |
| Coarse PPO adaptation | Seeds 0, 1, 2 at update 500 | 0/24 |
| Precision PPO adaptation | Seeds 0, 1, 2 at update 500 | 0/24 |
| Coarse world-model adaptation | Seeds 0, 1, 2 at update 500 | 0/24 |
| Combined adaptation | Seed 0 at update 500 | 3/8 |
| Combined adaptation | Seed 1 stopped at retained checkpoint 232; seed 2 not run | Not evaluated |

Combined seed 0 has useful partial evidence: terminal-window median hand position
errors 0.425/0.352 mm, orientation 0.153/0.111 degrees, aperture 0.058/0.161 mm,
and no recorded joint-bound/physics-warning failures. Its worst terminal position
errors are 0.993/8.316 mm and orientation errors 0.446/1.917 degrees; only 3 of 8
tasks meet the simultaneous continuous hold rule. Preserve this positive evidence
alongside failures. One seed and an incomplete cohort do not establish non-regression,
convergence, native-method performance, or promotion readiness.

`data/stationary-benchmark/vector-million-v1-interruption.json` records exact coverage
and checkpoint paths. All prior checkpoints and evaluation bytes remain unchanged.

## Reward audit and new experimental objective

The frozen common reward averaged narrow Gaussian kernels over scalar coordinates.
It can increase while required Euclidean pose error grows: changing one hand's error
from (20,20,0) mm to (35,0,0) mm worsens its norm from 28.284 to 35 mm but raises the
old precision reward from 0.859759 to 0.928572, with other terms held equal. Rotating
world axes also changes that score without changing physical pose mismatch.

The read-only TRAIN audit inspected 4,096 retained transitions from seed 0/update 500
of coarse PPO, precision PPO and coarse world-model cells. All position coordinates
at least 5 cm from their goals had reward derivatives below 1e-6 per metre; medians
were zero, zero and 2.608e-20. Recomputed rewards matched recorded rewards within
1.8e-7. This is objective-landscape evidence, not a measurement of PPO optimizer
gradients: the full projected native-q gradients were not zero. No evaluation
transitions or new optimizer steps were used. Reproduction, checkpoint hashes,
counterexamples and complete mechanism inventory are under
`data/stationary-benchmark/objective-audit-v1/{audit.py,results.json,REPORT.md}`;
the helper explicitly imports the preserved original checkout.

Correction `cd3eb346da11388ed26d5feaceacc9628fa7bd97` ports the pinned
[PolicyIK tracking and precision terms](https://github.com/Tinaliu0123/PolicyIK/blob/22f3a5cddc674e88995cae18b93cd5a5cb3ddb86/simulation/policyik_sim/tasks/env_cfg.py),
including inherited OpenArm position terms. For each required pose it sums
`-.2*d + .1*(1-tanh(d/.1)) + (1-tanh(k/.20)) + (1-tanh(k/.05))`, where `d` is
Euclidean position distance and `k` is mean distance between three centred 45 cm
virtual keypoints. These are reward features, not new robot geometry. Source 1 cm
is an observation scale, not a tracking kernel width.

Aperture uses an explicitly new one-dimensional extension of the source position
pair, without fictitious keypoints. Precision uses the exact source utility improvement,
discounted potential, capped backslide, binary dwell and joint-margin expressions,
with a declared global maximum tolerance ratio across required poses/apertures.
Transition rewards carry prior state; first-step/changed-goal history resets explicitly.
Source position/angular gain thresholds are preserved; aperture extends the position
ramp. All reward terms multiply actual control time (50 Hz here versus 30 Hz upstream).

The correction passed 37 focused tests, including independent transformed-keypoint
oracles, rotation invariance, far-field descent, scalar/vector/projected parity and
exact interrupted PPO/world-model continuation. A follow-up found that JAX's scalar
`abs` assigns an unwanted derivative at exactly zero aperture error. The follow-up
uses the same safe scalar norm for aperture tracking and tolerance ratios, with
zero-gradient and two-sided descent regressions for affine/curved owner laws.
Follow-up `455390a` passed the same 37-test focused batch. Neither test count is a
new cohort score. A TRAIN-only stability pilot (`reward-pilot-v3`) then completed
10 updates × 64 environments × 32 steps for each of the four learned adaptations:
20,480 real transitions per method, with finite reported objectives and losses. It
contains no development/final evaluation and establishes no accuracy gain.

The separately authorized `vector-million-reward-v3` cohort completed from frozen
checkout `455390a` in 1,728.653 seconds, including training, checkpointing and
development evaluation. Each learned seed used the same 500-update budget
(1,024,000 real transitions), three seeds per method, eight common development
tasks per seed, and unchanged simultaneous final-second held-pose thresholds.
Final-confirmation seeds remain reserved.

| Method | Strict held tasks | Constraint-failure tasks |
| --- | --- | --- |
| Numerical IK | 8/8 | 0/8 |
| Coarse PPO adaptation | 0/24 | 24/24 |
| Precision PPO adaptation | 15/24 | 0/24 |
| Coarse world-model adaptation | 0/24 | 0/24 |
| Combined adaptation | 12/24 | 0/24 |

The report decision is `do_not_promote`. Paired descriptive combined-minus-precision-PPO
held failure is +0.125 (12.5 percentage points worse). Mean per-task terminal-window
95th-percentile position error differs by +0.400 mm on the left and -0.323 mm on the
right. Those mixed directions and the held-task deficit do not establish
non-regression. The intervals are descriptive paired training-seed/task-family
bootstraps, not final confirmation. This controlled adaptation result is not a native
upstream score, convergence proof, or physical qualification.

Complete result/report: `data/stationary-benchmark/vector-million-reward-v3/`.
Two retained visualizations show **only task 0, training seed 0**, using actual
recorded native coordinates and requested URR frames:

- `data/stationary-benchmark/vector-million-reward-v3-precision-replay/`: DLS and
  precision PPO, with `replay.mp4`, contact sheet, error curves and source receipt.
- `data/stationary-benchmark/vector-million-reward-v3-comparison-replay/`: precision
  PPO and combined, with the same outputs. This selected view does not replace the
  complete 104-task cohort records.

New cohort publication preparation is separate from the verified first-screening
pack above. Publisher inventory v2 accepts explicit replay directories, checks their
body/scene/task receipt identities, retains the publisher code hash, and treats the
full historical checkout as canonical even without a supplemental `source/` snapshot.
It requires a new output directory for a v1→v2 transition; old pack bytes, inventory,
receipts and readback remain unchanged. No new cohort publication is claimed here.

Remaining adaptations are explicit: two-arm/aperture task, nominal MuJoCo and prepared
native action scales/governor, noiseless world-frame observation superset, canonical
native bounds instead of upstream arm soft limits, normalized action-rate cost,
small separate networks, Adam, tanh Gaussian exploration and finite-horizon
world-model gradients without native XMimic's critic bootstrap. Radian-only velocity
penalties were not guessed for prismatic coordinates. Native baselines stay unchanged.
Weighted source rewards can still trade errors; strict per-role held-pose acceptance
remains independent. Any corrected training requires a new frozen source/request/run;
do not rewrite, resume under new code, or draw upstream-failure conclusions from the
old prototype.

## Structured-model correction retained

The first world model predicted whole observations, including known goals and
kinematic error channels. The structured follow-up learns only native q/qd changes.
It preserves goals, derives command integration/rate limits, computes canonical
FK, and derives aperture/error/history channels analytically. Known geometry is
not a network output with authority over the reward. This is a new experiment,
not a relabelled old result. Its source is `3f3e089`.

The existing differentiable Kinematics now supports scalar hinge/slide/declared
mimic metadata. The original metadata path returns bitwise-identical outputs on20
nonzero batched states. Actual PiPER projected observations agree within2.21e-6
and aperture within5.6e-9 m; gradients at exact goals are finite. A one-update
actual-physics smoke completed, reloaded its checkpoints, and reported zero warnings.
These establish projection/integration behavior, not a learning gain.

Structured training currently refuses moving goals because trajectory phase/future
state is not yet part of that learned state. The real feedback evaluator supports
moving targets, with separate timing and held-window results. Future final-test
seeds10000–10063 are reserved before tuning and must not enter training or validation.

## Commands

Use the exact owner source/archive and compatible shared actions/geometry/contracts
first on PYTHONPATH; do not repoint the shared development environment. Set
`SX_EMBODIMENTS_ASSETS` to the prepared closure for later portable consumption.

```bash
python tools/prepare_stationary_benchmark.py --output NEW_PREPARED
python tools/benchmark_urr.py --prepared NEW_PREPARED --output NEW_RUN \
  --methods dls ppo_coarse ppo world_model combined --seeds 0 1 2 \
  --evaluation-seeds 0 1 2 3 4 5 6 7 --iterations 20 --rollout-steps 128 \
  --episode-steps 250 --hold-steps 50
python tools/report_urr_benchmark.py --run NEW_RUN --output NEW_RUN/screening-report.json
python tools/render_urr_benchmark.py --prepared NEW_PREPARED \
  --task DLS=NEW_RUN/dls/seed-0/evaluation/task-0.json \
  --task Combined=NEW_RUN/combined/seed-0/evaluation/task-0.json --output NEW_VIDEO
python -m pytest -q tests/test_kin_scalar.py tests/test_urr_benchmark.py \
  tests/test_urr_learning.py tests/test_urr_reward.py tests/test_urr_vector.py \
  tests/test_stationary_benchmark.py tests/test_urr_benchmark_report.py
```

Reopening the same run checks its exact request and attaches to complete numeric
checkpoints. Changed code/configuration needs a new run directory; use the retained
historical checkout to resume an old experiment. Reports cannot promote a small
screening, and native scores remain a separate prerequisite. See the
[experimental contract](urr-controller-benchmarks.md),
[native baseline runtime/protocol](native-baselines.md), and
[managed execution](managed-training.md).
