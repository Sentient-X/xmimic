# Nominal Sentient simulation and SOMA retargeting

`examples/sentient_nominal.py` generates an explicit development experiment for the
39-coordinate Sentient body. `tools/prepare_embodiment_job.py` is the generic input
preparer: it accepts another supported canonical body with that body's explicit
nominal simulation profile, retarget recipe and training profile. Neither tool
submits jobs, admits training data, downloads motion archives or operates hardware.

## Exact body and portable ownership

The example binds body identity
`a2ad7e6e3f2b0cf5a73f1b1255f2e022206a057481ba3b8cb384066b682834b3`
and URDF SHA-256
`a999ac459a8214ae4d70c213917ae337583c6e9e03bd4be94d7630debe213cd3`.
It refuses a changed body until its experimental assumptions are reviewed.

The generic exporter belongs to `sx_embodiments.nominal_mujoco`. Portable decoding,
retarget target preparation and full nominal export of this 39-coordinate document
were tested against exporter commit
`96aabf07e16427d2c8e0006984cdd8d6d7e2d7cc`, based on
`93141cd990c5347e2985801dda49d9f5de4bd535`. The corrected body is supplied as its
canonical document and verified asset closure; this does not claim that the older
registry definition at that base revision is the corrected body. Worker images must
ship the owning exporter commit. A laptop-only module overlay is not a release.

## Explicit experimental assumptions

- The canonical URDF geometry and inertials are preserved. Their existing estimates
  remain estimates; no dynamics measurements are invented.
- PD gains, effort bounds, armatures and action scales are experimental choices by
  component role. They do not establish actuator ratings or hardware control gains.
- The waist rod is held at zero. Torso, waist, neck and fingers have explicit holds
  during this geometric retarget experiment. The nominal simulation starts at the
  nearest bounded zero; IK uses a small knee/hip bend as its initial seed.
- The two `foot2` links are designated supports. Their lowest collision vertices
  give approximately 49.5 mm sole offsets. The initial root height of approximately
  0.68784 m gives the lowest designated sole 10 mm clearance above the plane.
- Source centimetres become metres. The source-to-world basis and first-frame
  root/foot orientation calibration are explicit **nominal correspondences**.
  They are not inferred physical calibration facts.
- Target limb lengths and shoulder/hip anchors come from canonical target FK.
  Source upper/lower limb vectors are normalized to those lengths. BVH translation
  channels replace `OFFSET`, so the source's actual first-frame positions provide
  the source leg-length measurement used for root translation scaling.
- The recipe tracks wrist/ankle positions and calibrated ankle orientations.
  Collision, balance, foot planting, force feasibility and physical qualification
  remain separate checks.

The example is clip-calibrated. Its initial source posture is a declared calibration
choice, not a universal anatomical rest pose or a proven recipe for the full corpus.
Thresholds stay at 2 mm position residual and 0.01 rad orientation residual.

## Prepare private inputs

Run from the XMimic checkout with its dependencies and the pinned owning exporter
installed. Set `SX_EMBODIMENTS_ASSETS` to the verified portable canonical asset tree
when the body is not installed in the local asset store.

```sh
python -m examples.sentient_nominal \
  --embodiment-document /private/body/embodiment.json \
  --source /private/source.bvh \
  --output /private/experiment

python -m tools.prepare_embodiment_job \
  --embodiment-document /private/body/embodiment.json \
  --source /private/source.bvh \
  --nominal-profile /private/experiment/nominal-profile.json \
  --recipe /private/experiment/recipe.json \
  --training-profile /private/experiment/training-profile.json \
  --source-entry datasets.bones-seed.2f59b2077b9d \
  --output /private/prepared-job
```

The preparer copies the complete verified owning asset closure, exports the nominal
model, derives XMimic metadata, and records source/body/profile identities. The
source entry is a declared lineage reference; Catalog verifies authority and lineage
when it admits the managed inputs. Preparation alone does not establish feasibility
or training eligibility.

The managed CPU worker consumes these files and the explicit metadata-resolved
source family. Successful geometry produces a `training-candidate.json` transform
manifest whose eligibility is inherited from the input. Rejected clips retain all
candidate coordinates and residuals, and produce no training candidate. Catalog's
training-use admission remains authoritative for source ancestry and held-out data.

## Observed complete-clip result

The first retained SOMA-uniform BVH at dataset revision
`2f59b2077b9da34dd4e43618e705c7cb962c9a66` has source SHA-256
`a81ff5609be1bd6f1733059740a4116f7c630326331141335f42b1e1544e93b2`.
Its literal frame time is `0.008333`, preserved without relabeling it 30 Hz.
No licensed BVH bytes or actor-specific source paths are committed here.

| Check | Observed result |
| --- | --- |
| Explicit eight-frame preliminary check | 8/8 feasible; 1.805 mm maximum position error; 0.004094 rad maximum orientation error; 0.343 s CPU |
| Complete original clip | **Rejected**; all 599 frames retained; 423 feasible, 176 rejected; 86.45 s CPU |
| Complete-clip maximum position residual | Left wrist 9.419 mm; right wrist 23.863 mm; left ankle 2.603 mm; right ankle 1.126 mm |
| Complete-clip maximum orientation residual | 0.007526 rad |
| Complete-clip training output | No `motion.npz`; no training admission |

The short preliminary result did not establish whole-clip success. The full run did
not cut failed frames or loosen tolerances. These figures establish geometric
retarget behavior and retained rejection evidence, not a trained or qualified robot.

## Recovery artifacts

Managed single-GPU training checkpoints include complete carry and randomized
constants. Intermediate checkpoints use the managed runtime's persistent checkpoint
mechanism. Final output additionally contains `recovery/index.json` and bounded
parts of the opaque native state archive, alongside the inference policy.

`xmimic.managed_worker.restore_recovery_archive(directory, destination)` verifies
chunk membership, order, sizes, per-chunk SHA-256 and complete-archive SHA-256.
Reconstructing this archive is distinct from automatic runtime reattachment and
still requires native configuration/body/source binding and pytree-template checks
before training can continue. Policy NPZ files alone are not resumable training state.

### Explicit nominal reference contract

The original 599-frame solve remains rejected by its 2 mm optimization goal. A separately versioned nominal simulation contract now accepts at most 30 mm positional approximation and 0.01 rad orientation error while leaving that optimizer goal unchanged. This bound is an experimental motion-tracking choice, not a measured robot capability.

`NominalQuality` retains every native joint sample and original source timebase in `candidate.npz`. It translates each whole pose vertically until the lowest designated foot collision geometry is 2 mm above the floor; a correction above 50 mm refuses. All robot collision geometry must remain above -1 mm. The reference clock is explicitly dilated by four, and the resulting discrete joint velocity must stay below 6 rad/s. Native coordinate bounds remain mandatory. The original optimizer result, original candidate, per-frame vertical corrections, new owner scene/profile hashes, and new reference timebase remain in the derived cache. No foot-contact labels, no-slip condition, balance result, or physical qualification are inferred.

The owner exporter now records `CollisionMode.GROUND_ONLY` in this example's nominal profile: robot collision geometry has `contype=2, conaffinity=1`; the floor has `contype=1, conaffinity=2`. This avoids treating overlapping convex mesh approximations of fixed camera/shoulder structure as a qualified self-collision model. Source meshes, body identity, inertials, native limits, and hardware facts are unchanged. Legacy profiles without this field retain full collision behavior.

The retained real SOMA candidate passed this nominal contract with 599/599 frames, maximum root correction 21.987 mm, minimum collision-geometry floor height 2 mm, maximum joint speed 4.7612 rad/s, maximum positional residual 23.863 mm, and maximum orientation residual 0.007526 rad. Its source rate remains 120.004800192 Hz; the reference is 30.001200048 Hz. This is reference preparation, not a trained-policy or physical-robot result.

To reuse retained solver work without repeating IK:

```bash
python -m tools.qualify_retarget --source-cache /private/original-cache/key \
  --recipe /private/nominal/recipe.json --robot /private/nominal/robot \
  --cache /private/derived-cache --execution-ref retained-nominal-projection
```

The command refuses changes to source mapping or optimizer settings, verifies the original completed cache, and writes a distinct derived cache. The old rejection is immutable. `tools/render_retarget.py` renders source and reference with actual MuJoCo robot meshes, residuals and native joint-limit indicators; the video displays both source time and reference time. It is a kinematic replay, not a physics rollout.

**Engineering hold (2026-11-08):** the user rejected the SO-101-headed development body for further learning promotion and requested restoration of the earlier humanoid head with human engineering review. The measurements above are preserved diagnostics of that superseded candidate, not authorization to train it or a qualification of its construction. No GPU training was performed for that candidate. The generic parser, projection, recovery and visualization mechanisms remain reusable after a reviewed body is selected.
