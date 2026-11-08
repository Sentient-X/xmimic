# Stationary URR controller screening

Status: the benchmark runner executes actual nominal MuJoCo physics. The first
screening is complete; a structured-model follow-up is running. Native vanilla
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

## Controlled correction under test

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
  tests/test_urr_learning.py tests/test_stationary_benchmark.py tests/test_urr_benchmark_report.py
```

Reopening the same run checks its exact request and attaches to complete numeric
checkpoints. Changed code/configuration needs a new run directory; use the retained
historical checkout to resume an old experiment. Reports cannot promote a small
screening, and native scores remain a separate prerequisite. See the
[experimental contract](urr-controller-benchmarks.md),
[native baseline runtime/protocol](native-baselines.md), and
[managed execution](managed-training.md).
