# Whole-body retargeting and how to judge it

`xmimic/retarget_whole_body.py` turns a human skeleton clip into a reference motion for any
articulated body whose embodiment document names semantic sites (`pelvis`, `chest`,
`left_shoulder`, `left_elbow`, `left_wrist`, `left_hand`, `left_hip`, `left_knee`,
`left_ankle`, `left_toe_edge` and the right-hand counterparts). Nothing in the code names a
robot, a limb or a joint. The source declares one table: semantic site to source joint
(`tools/soma_correspondence.json` for BONES-SEED SOMA, plus its source-to-world axes).

## What is derived from the body

| Quantity | Derived from |
| --- | --- |
| Site tree | nearest corresponding site on each site's kinematic ancestor path |
| Rigid vs articulated segment | child site fixed in the parent site's frame for random joint values |
| Solved / held joints | ancestors of any corresponding site are solved; the rest are held at the reference state |
| Orientation offsets | one direction-only fit of the body to a declared human calibration frame |
| Segment vectors | the robot's own segment vector at calibration, carried by the human bone that ends at the child joint |
| Vertical scale | robot root height above its lowest body point / human root above its lowest joint |
| Orientation targets | every site except a rigid child that starts an articulated segment (its link carries only part of a ball joint) |

Calibration fits articulated segments (weight 1), rigid leaf pairs (hands, feet; they fix limb
twist) and rigid siblings (lateral axes) at weight 0.5, with a 0.2 prior toward the reference
state. Rigid pairs inside the trunk are left out: on the Sentient body they disagree with the
human by 23-33 degrees purely because of geometry.

## Per-frame solve and post-processing

Bounded least squares (SciPy TRF) over the solved dofs: site positions (root weighted 3x),
site orientations (root 1.0, others 0.2), an acceleration prior (deviation from the
constant-velocity prediction) and a weak posture prior. Joint limits are hard bounds; a
25 rad/s per-frame joint speed bound turns IK branch switches into bounded transitions.
Each frame first places the floating root exactly on its target pose, so fast rotations
(flips) never accumulate root drift. Then a 12 Hz zero-phase low-pass and a ground pass:
the lowest body mesh point follows the scaled human's lowest joint (never below the floor).
The source BVH `Root` joint (a ground projection) is excluded from the human lowest point.

## Dynamically consistent flight (`xmimic/retarget_flight.py`)

A kinematic retarget copies the human's flight; with the robot's own masses that flight is not
physical (on ca3fb7 the flight CoM was 0.07-0.12 m off a gravity parabola and the angular momentum
about the CoM drifted 20-43 %), and the batch-1 policy fell at flight on every flip. The step keeps
the joint trajectory and rewrites only the floating root inside each flight window (lowest body
point above 3 cm for at least 80 ms): root orientation is integrated so the angular momentum about
the CoM is constant (its axis from the reference mean, magnitude and direction searched so the
landing orientation matches), then the root translation puts the CoM on the ballistic parabola
between takeoff and landing CoM; 50 ms blends at both ends. If the needed takeoff vertical speed
exceeds the bound sum_i |d(CoM - support)_z/dq_i| * rated speed_i, the apex is lowered and the
flight shortened to 2 v/g-consistent duration (time warp). That bound ignores torque: an upper bound.

```bash
$PY -m tools.flight_consistency --body-document BODY.json --scene SCENE.xml \
  --correspondence tools/soma_correspondence.json --ratings RATINGS.json \
  --speed-override DM10010=10.47 --joint-names META --embodiment-id ID --output OUT_RUN IN_RUN/<clip> ...
```

Observed on the 8 flips (2026-11-16): CoM residual 0.07-0.12 m -> 0 (flight interior), angular
momentum drift 0.21-0.43 -> 0.03-0.09 on the main flights, landing orientation error <= 1.5 deg;
needed takeoff 1.9-2.7 m/s vs bound 3.4-4.6 m/s (48 V DM10010), so no time warp; side-view keyframes
still read as the same flips. Mid-flight root pitch now differs from the human's (different inertia).
Published as `xmimic-runs:motions/demo-motions-v3`.

| Failure | Cause | Fix |
| --- | --- | --- |
| Angular momentum all zero / singular solve | `mj_subtreeVel` needs `mj_comVel` first | Call kinematics, comPos, comVel, subtreeVel |
| Landing 150 deg off after Gauss-Newton on the rotation vector | Rotation-vector residual wraps for a 280 deg flip | Bounded scalar search on the mean-L magnitude, then Nelder-Mead on 3 components |

## Judging harness

`tools/package_retargets.py --flat --ratings RATINGS.json` writes the trainer's flat directory (one
`<clip>.npz` each, manifest beside it, not inside) and per joint the peak speed over the actuator's
rated and no-load speed (`speed_over_rating`, `joints_over_no_load`).

`tools/judge_retarget.py` writes, per clip, under `<clip>/judge/`:

- `keyframes.png`: phase keyframes (takeoff, most inverted, apex, landing; otherwise limb
  extremes and deepest crouch) x four views (front, side, top, tracking 3/4). Robot links
  take their segment's colour (left warm, right cool, trunk grey; side derived from the
  calibration pose) and the scaled human (the solve targets) stands beside it in the same
  colours, so a swapped side or a wrong-way knee shows as a colour mismatch. Root trace in white.
- `keyframes_overlay.png`: the same views with a 35 % transparent robot over a thin skeleton.
- `summary.png`: segment direction error heatmap (segments x time), joint-at-limit map,
  10 fastest joints, 10 largest inverse-dynamics torque proxies, root height / lowest point /
  tilt with keyframe markers, root external force (body weights) and self-proximity flags.
- `sidebyside.mp4`: real-time tracking 3/4 video, robot beside the scaled human.

`xmimic/retarget_judge.py` declares a numeric screen before results are seen (direction p99
15 degrees, max 35, floor penetration -1 cm, contact slide p95 0.3 m/s, joint jitter above 8 Hz
0.5 degrees). It is never widened afterwards; acceptance is a recorded visual judgment.

```bash
PY=~/.cache/retarget-venv/.venv/bin/python   # mujoco, scipy, matplotlib, pillow, imageio-ffmpeg
N=data/bones_seed/nominal-approved-ca3fb7
COMMON="--body-document $N/toe-sites-v1/embodiment.json --scene $N/input/robot/scene.xml \
  --calibration-source $N/source.bvh --correspondence tools/soma_correspondence.json"
$PY -m tools.retarget_clips $COMMON --embodiment-id <id> --joint-names $N/input/robot/meta.json \
  --output RUN_DIR CLIP.bvh ...
$PY -m tools.judge_retarget $COMMON RUN_DIR/<clip> ...
# body facts stay in the embodiment layer; a retarget may only narrow them, declared in a file:
#   --retarget-limits LIMITS.json   {"ranges": {"left_elbow_pitch_joint": [-1.49, 0.0], ...}}
$PY -m tools.package_retargets --judgments JUDGMENTS.json --output OUT --volume xmimic-motions
```

## Evidence on BONES-SEED (2026-11-16, body ca3fb7)

Same 599-frame nominal clip as the rejected URR run, same raw metric (robot site pair vs source
joint pair): max 31.6 degrees here vs 63.3 degrees for `urr-profile-v3`. Calibration residual
max 11.0 degrees (leg site pairs sit on link origins offset from the bone line).
Solve 50-130 s per 500-1000 frame clip on one CPU core; judging about 2.5 min per clip.
22 clips judged, 18 accepted (final set on `xmimic-runs:motions/demo-motions-v1`) (4 backflips, 2 front flips, 2 side flips, 2 cartwheels, dances,
floor scissor kick, broad jump, boxing, high knees). Every clip fails the 15 degree p99 screen;
the dominant errors are arm segments at body joint-range limits (see below).

| Failure seen | Cause | Fix / status |
| --- | --- | --- |
| Hip yaw pinned at +-90 deg in every clip | Calibration fit only articulated pairs; thigh twist unobserved | Rigid leaf pairs (feet, hands) and a reference-state prior in calibration |
| Calibration 28-33 deg off, torso tilted | Rigid trunk pairs (chest->shoulder, pelvis->hip) differ by body geometry | Leave rigid non-leaf pairs out of calibration |
| Robot pulled to the floor mid-backflip (shift -0.75 m) | BVH `Root` joint sits on the floor and counted as the human's lowest point | Lowest point over the root site's source subtree only |
| Front flip diverged, root at 300 m | Root drift accumulated through the inversion | Anchor the root to its target pose before every frame |
| 80 rad/s one-frame arm pops | IK branch switch in one frame | 25 rad/s per-frame speed bound |
| Limb re-seed experiment | Rarely triggered (errors < 6 cm), and the one switch it made produced an 81 rad/s pop | Removed |
| Synthetic test joints frozen | MJCF without `angle="radian"` reads ranges in degrees | Test body declares radians; real scenes already do |
| Elbows stuck at limit 50 % of the calm nominal clip | Real hardware: q=0 is the elbow's forward stop at 90 deg flexion (embodiment owner, FK + teleop history); the range continues ~80 deg past straight | Flexion beyond 90 deg clamps at the stop. The swing past straight is never used: `--retarget-limits` (narrowing only, refused if it widens) holds left [-1.49, 0], right [0, 1.47]; 10 of 18 clips had used it, up to 0.5 rad |
| Claimed right_hip_roll range was unmirrored (wrong) | Read `jnt_axis` in the joint's local frame; in world the right axis is -X, so +roll abducts and [-0.28, 1.92] is correct | Always judge a range by FK of a distal point, never by local axis + numbers. A declared mirrored override made the cartwheel worse (right leg pinned at the new limit, 45 deg) and was withdrawn; the arm-yaw "offset" claim had no FK and was retracted |
| Mirrored cartwheel legs do not split | Genuine 16 deg adduction limit on the leading leg plus the elbow cap | Rejected; the jumping jack, first rejected from the 3/4 view, matches in the front view and was accepted |

Published sets are immutable: on 2026-11-16 an elbow fix overwrote ten files of
`motions/demo-motions-v1` while a paid training run read that path. Fixes go to a new version
(`demo-motions-v2`); the packager now refuses an existing destination.

Stop conditions: a changed body identity means re-running every clip; never widen the screen
after seeing results; never crop silently (`--window` is recorded in metrics.json). BVH, motions
and renders are private (BONES-SEED license): `data/` is git-ignored.
