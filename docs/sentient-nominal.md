# Nominal Sentient simulation and SOMA retargeting

The owner approved the restored-head, restored-shoulder assembly for **nominal
simulation testing** on 2026-11-08. The experiment uses 33 controlled coordinates,
explicit nominal servo assumptions and ground-only collisions. It does not supply
measured actuator facts, physical qualification or a trained policy.

## Exact inputs

- Body: `ca3fb7c3b69a4df643d21904d64b57c7d954c536dbb70660143d75133e53f317`.
- URDF: `e7fd42a06794c5d3fce549e55630ea0a9194eee04411b7f06a9d7ed5ada87e67`.
- Owning source snapshot: `a5528ec576615cde3e0c6645b2a4fd548b72e942`, combining
  assembly `a01160b` with the corrected v2 exporter from `5780529`.
- Nominal profile: `333ed58a2cf7720b3c4afeaf80ca43ca6ff584a7d82d26774c5c1323ba87cf1a`.
- Scene: `c691cdbb034c4408854cbd14641c7fad22657f2c19b1c95a508e744e7f86c542`.
- Complete SOMA source: `a81ff5609be1bd6f1733059740a4116f7c630326331141335f42b1e1544e93b2`,
  599 frames, literal frame time `0.008333`, from the privately mirrored BONES-SEED
  revision `2f59b2077b9da34dd4e43618e705c7cb962c9a66`.

The body and source are portable, hash-bound inputs. Neither a registry alias nor
an installed older embodiment package is a substitute. Exporter `5780529` alone
predates the restored shoulder assets; use the coherent owning source above.
Runtime images still require publication with compatible owning source.

## Explicit assumptions

The source URDF geometry and inertials remain intact, including their documented
estimates. Experimental gains, effort bounds, armatures and action scales are
chosen by component role. They are not hardware ratings or calibrated gains.
The waist rod, neck yaw and head pitch are explicitly held at zero; no original-head
drive binding is invented. The initial pose is bounded zero, not a calibrated
standing pose. The IK seed uses a small knee/hip bend.

The two `foot2` links designate support geometry. Source centimetres become metres;
source axes and first-frame root/foot correspondence are explicit nominal choices.
Canonical limb lengths normalize the source segments. Each clip's first posture is
a declared calibration choice, not a universal recipe for the entire corpus.

The optimizer still targets 2 mm and 0.01 rad. A separately declared nominal
reference contract permits up to 30 mm position error, 0.01 rad orientation error,
6 rad/s joint speed and a 50 mm whole-pose vertical correction. It puts the lowest
support geometry 2 mm above the floor and requires all collision geometry above
-1 mm. The reference clock is slowed by four; original coordinates, timebase,
residuals and per-frame corrections remain retained. Self collisions are disabled
explicitly. No foot-contact labels, no-slip or balance guarantee is inferred.

## Reproduce preparation and diagnostics

Install the exact owning source and set `SX_EMBODIMENTS_ASSETS` to the verified
portable asset tree. From XMimic:

```sh
python -m examples.sentient_nominal \
  --embodiment-document /private/body/embodiment.json \
  --source /private/source.bvh --output /private/profile

python -m tools.prepare_embodiment_job \
  --embodiment-document /private/body/embodiment.json \
  --source /private/source.bvh \
  --nominal-profile /private/profile/nominal-profile.json \
  --recipe /private/profile/recipe.json \
  --training-profile /private/profile/training-profile.json \
  --source-entry datasets.bones-seed.2f59b2077b9d --output /private/input

python -m tools.diagnose_nominal \
  --robot /private/input/robot --output /private/pd-baseline --seconds 2
```

The diagnostic saves every physics state, controls, forces, contact distances,
warnings and input hashes before rendering. It uses constant owner-profile position
targets and never tunes away a fall. It is not a learned controller. Use new derived
output directories; the tool refuses existing outputs and paths under `raw/`.

`xmimic.retarget_workflow.retarget_clip` is the shared local/managed CPU solver.
Give it the exact body, source, recipe, robot bundle, source-family identity,
execution reference and `ArtifactCache`; every frame is durably accounted for.
The [managed workflow](managed-training.md) publishes and schedules it through the
platform. To inspect one completed cache:

```sh
python -m tools.render_retarget \
  --source /private/source.bvh --recipe /private/input/recipe.json \
  --robot /private/input/robot --cache /private/cache/EXACT_KEY \
  --output /private/reference-visualization
```

This second video is kinematic reference playback, not physics. Its joint count
comes from the selected embodiment. The physics baseline and reference playback
are separate artifacts with explicit labels.

## Observed result on the approved body

| Check | Result |
| --- | --- |
| Owner FK versus exported model, all 59 links at nonzero pose | Position error ≤3.33e-16 m; rotation matrix error ≤9.99e-16 |
| Two-second CPU physics, 400 steps | Finite throughout; zero MuJoCo warnings; contacts exercised |
| Constant-position PD standing | **Failed**: root height 0.688→0.289 m; robot falls forward |
| Peak contact penetration in that baseline | 28.2 mm |
| Complete source retarget | 599/599 frames retained; 88.76 s CPU |
| Original 2 mm optimizer goal | Not satisfied for the whole clip |
| Predeclared nominal reference contract | Accepted all 599 frames |
| Maximum landmark / orientation error | 23.863 mm / 0.007526 rad |
| Maximum vertical correction / joint speed | 21.983 mm / 4.7612 rad/s |
| Minimum reference collision-geometry height | 2 mm |
| Original / reference sample rate | 120.004800192 / 30.001200048 Hz |

The retarget cache key is
`7f2d793d4a9e66eb5171498c2c7d8698d14c686e88cdb87fe065ef46132d5233`;
manifest SHA-256 is
`04f2c340dc1402bb41b09c3904588988c7f77facc54e5754eb95498b6483a624`.
Private inputs, receipts and videos are retained under
`data/bones_seed/nominal-approved-ca3fb7/`. Catalog training admission remains
separate from local reference acceptance.

The complete cache was published privately through Catalog to R2, linked to the
mirrored BONES-SEED source entry, then all four members were downloaded and verified
against the manifest above. Its pack is `xmimic.retarget.` followed by the cache key.
It remains marked `training_admitted=false`; storage and source lineage do not grant
learning access or establish balance. No Hugging Face download was repeated.

The earlier 39-coordinate SO101-head body is superseded. Its old nominal exporter
used intrinsic `xyz` instead of URDF's extrinsic `XYZ`; its floor/collision/render
conclusions are invalid. Old caches remain diagnostic history and cannot be reused
under this new body identity. No GPU training used those artifacts.

The generic platform PR remains unpublished: required protocol-tool downloads
return HTTP 504 before CI checks run. No Modal CPU/H100 job, trained policy,
hosted recovery result or physical operation is claimed by these local diagnostics.
