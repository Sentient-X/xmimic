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

## Judging harness

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
$PY -m tools.package_retargets --judgments JUDGMENTS.json --output OUT --volume xmimic-motions
```

## Evidence on BONES-SEED (2026-11-16, body ca3fb7)

Same 599-frame nominal clip as the rejected URR run, same raw metric (robot site pair vs source
joint pair): max 31.6 degrees here vs 63.3 degrees for `urr-profile-v3`. Calibration residual
max 11.0 degrees (leg site pairs sit on link origins offset from the bone line).
Solve 50-130 s per 500-1000 frame clip on one CPU core; judging about 2.5 min per clip.
22 clips judged, 17 accepted (4 backflips, 2 front flips, 2 side flips, 2 cartwheels, dances,
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
| Elbows stuck at limit 50 % of the calm nominal clip | Body model: elbow q=0 is 90 deg flexed, range allows 0-90 deg flexion and ~80 deg hyperextension | Body range bug reported to the embodiment owner; not compensated here |
| Mirrored cartwheel / jumping jack legs do not spread | right_hip_roll range [-0.28, 1.92] is the left range unmirrored: 16 deg abduction | Body range; clips rejected until corrected |

Stop conditions: a changed body identity means re-running every clip; never widen the screen
after seeing results; never crop silently (`--window` is recorded in metrics.json). BVH, motions
and renders are private (BONES-SEED license): `data/` is git-ignored.
