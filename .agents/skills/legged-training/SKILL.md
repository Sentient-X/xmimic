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

After composing the corrected full-body candidate, the same audit reported 39
coordinates, seven per arm, eight shared components with equal part facts (arms,
grippers, neck, cameras and IMU), and only the retained torso with different facts.
There is no active MJCF on the corrected candidate; historical simulation bytes
remain historical. Shared arm ready references do not qualify a whole-body pose.

From the owning `sx` checkout, the corrected full-body candidate passed 262 tests
in 44.81 seconds with:

```bash
PYTHONPATH=packages/sx-embodiments .venv/bin/python -m pytest \
  packages/sx-embodiments/tests/test_successor.py \
  packages/sx-embodiments/tests/test_static_upper_body_modules.py \
  packages/sx-embodiments/tests/test_mechanical.py \
  packages/sx-embodiments/tests/test_static_upper_body.py \
  packages/sx-embodiments/tests/test_recorded.py \
  packages/sx-embodiments/tests/test_driver_bindings.py \
  packages/sx-embodiments/tests/test_render_packs.py \
  packages/sx-embodiments/tests/test_humanoid.py \
  packages/sx-embodiments/tests/test_humanoid_correction.py \
  packages/sx-embodiments/tests/test_assembly.py \
  packages/sx-embodiments/tests/test_known.py \
  packages/sx-embodiments/tests/test_layout_laws.py \
  packages/sx-embodiments/tests/test_asset_integrity.py \
  packages/sx-embodiments/tests/test_model_laws.py -q
```

The owning manifest check passed against tracked assets plus the new composite;
176 pre-existing untracked files in the shared checkout were excluded, not silently
admitted. Pyright retained 12 existing `media.py` errors and none in changed files.
These checks used the existing pinned baseline, not newer `main` or a live station.

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
| Full humanoid misses upper-body corrections | Two source definitions, not one shared corrected module | Reuse corrected modules at existing full-body seats; preserve lower-body facts and explicitly derive different-frame placements |
| A fitted-module refactor changes joint positions in vectors | Native order follows nested component ownership | Retain exact historical document; regenerate bindings; never alias old identity to new order |
| Corrected parts' relative meshes resolve in the old humanoid directory | The composed description contains parts from different asset directories | Resolve relative mesh URIs against each member's source in the owning composer |
| A mixed DAS/static-body build cannot resolve `package://DAS_Gripper_urdf/…` after relocation | Preserving every ROS package URI loses vendor aliases that depend on the original source directory | Normalize all references through the source-aware owner; exercise foreign aliases as well as relative paths, and retain historical assets when normalization changes generated bytes |
| Asset manifest refuses a package declared both public and entitled | The new composite combines old BSD geometry and proprietary corrected geometry | Publish the composite under its own entitled asset directory; keep the old public description intact for history |

## Mirror source custody through Catalog

Use `tools/mirror_bones_seed.py` as documented in `docs/bones-seed.md`. Preconditions:
approved HF access, an authenticated complete repository-tree plan, and a Catalog
principal authorized to write a private project. The pinned snapshot is 541 files,
114,370,281,701 bytes; normalized inventory digest
`b1b79bbce35df8749740373218e107763e0c514b68adfd763d0d8d8d2929c220`.
Do not put the private plan, source bytes or credentials in git.

The owner clarified that G1-retargeted motions are irrelevant to training their
robots. Use SOMA human motion, shape parameters and metadata, then retarget
directly to the canonical target embodiment. The full repository snapshot also
contains the already-copied G1 reference archive; do not present its transfer as
a Sentient training milestone or route retargeting through G1. Verify the final
readback with a SOMA archive chunk. Human-source custody excluding G1 is
90,870,308,054 bytes.

Read the source itself before choosing a BVH parser. A bounded R2 prefix inspection
found translation channels below the root and rotating named terminal joints;
parsers that assume only the root translates or that every terminal is an inert
`End Site` lose source motion. Preserve literal frame time separately from nominal
metadata fps. Neither inspected header established units, handedness or up-axis.
The private `data/bones_seed/soma-header-inspection.json` records the two-header
sample and limits (262,154 compressed / 1,464,320 decoded bytes, five tar entries).
No motion rows or proportional skeletons were inspected.

On 2026-11-07 the transport failure suite passed 34 tests with the installed SDK:

```bash
python -m unittest discover -s tests -p 'test_mirror_bones_seed.py' -q
```

The live license-file smoke uploaded and read back 21,781 bytes through Catalog,
SHA-256 `eb9b70b084e0a5dcda8465a4be2ba2f5b81e3c3800305c727777a3176defbc5f`.
The subsequent full mirror completed with exit code 0 and every original-file
source hash verified. Human source excluding G1 is 90,870,308,054 bytes; the
complete repository receipt covers 541 files / 114,370,281,701 bytes. Manifest
SHA-256 is `7b97480d92b24790ce063697014307247514bb45b0fba80909a5096b25dbb3b5`.
Independent Catalog access matched all 647 expected members. With no HF token
and `HF_HUB_OFFLINE=1`, the license, metadata, base skeleton and one complete
1 GiB SOMA-uniform chunk restored with matching hashes. This sample does not
claim a second full-corpus readback. Training admission remains separate.

Rebuild the inventory from the restored parquet:

```bash
HF_HUB_OFFLINE=1 python tools/prepare_bones_seed.py \
  --metadata /private/catalog-restored/seed_metadata_v004.parquet \
  --output /private/r2-inventory
```

With PyArrow 24.0.0 this took 9.98 seconds on the existing transfer host and
returned 142,220 motions, 71,132 originals, 71,088 mirrors and zero unresolved
mirror parents. Keep the metadata-only inventory receipt alongside the separate
archive-custody receipt; neither retargeting nor training has run. Private evidence
is under ignored `data/bones_seed/r2-evidence/`.
Live checks also returned 401 for anonymous pack lookup/download access and 404
for representation lookup from another authorized project. The public API does
not return an `is_public` field; the owning registration path defaults it to false.

| Failure/signature | Established cause | Correction |
|---|---|---|
| Importing the checkout SDK fails on missing generated `AssetRead` | Shared editable SDK/protobuf state differs from the released SDK | Use an isolated environment pinned to the installed release; do not rewrite another session's generated tree |
| A noninteractive SSH session cannot find `uv` | The installed user binary is absent from that session's PATH | Resolve the installed executable explicitly; the transfer venv intentionally had neither pip nor PyArrow, so install the pinned PyArrow into that isolated environment |
| A 45 GB archive exceeds worker scratch | Whole-file promotion materializes the source on the Catalog worker | Use bounded 1 GiB blobs through Catalog staging, with an ordered exact-byte manifest |
| Upload succeeds but bytes cannot be read | Promotion alone does not confer entry-scoped access | Register private per-chunk packs, then the immutable final pack after full-source verification |
| HF download is very slow on the laptop | The measured route took 124.14 s for 8 MiB; an existing datacenter worker took 0.63 s for identical bytes | Run the bounded transfer client near R2 through the same Catalog API; do not copy cloud credentials or bypass the owner |
| Copying evidence back to the laptop is slow | Slow host-to-laptop transfer was observed; its network cause was not established | Run the offline inventory beside the Catalog-restored metadata; return compressed receipts with per-file hash checks rather than copying the parquet again |
| A normal Git file fails SHA-1 verification | Git hashes the blob header plus contents | Verify `blob <size>\0` plus bytes; use raw SHA-256 for LFS contents |
| A correctly hashed manifest restores zero files | Content hashing alone does not prove corpus completeness | Require the pinned normalized inventory, not self-declared counts |
| Restore repeats completed archive chunks | Opening the partial file with `wb` discarded progress | Bind the journal to manifest identity, rehash retained boundaries and append |
| Transfer stops with `CatalogUnavailableError` after acceptance | A live result read or pack-registration boundary became unavailable; the saved operation subsequently succeeded | Inspect that exact operation and API readiness; reattach reads with fresh clients, never resubmit paid work. The tool bounds read reattachment to two attempts; writes still stop for inspection |
| Multipart upload stopped after 14 recorded parts | The old failure discarded the HTTP status, so its precise cause remains unknown | Preserve the ETags and resume the same upload. Controlled status/part diagnostics now distinguish failures; only explicit 429/500/502/503/504 responses get two fresh-grant retries for identical bytes. Permission, integrity and unknown transport failures still stop |

Resume with the same output directory, project, pack and source plan. Inspect a
failed owner operation before resuming; never blindly replace its durable
reference. The tool prunes only its verified transport scratch and refuses scratch
under `raw/`. Preserve source licensing, private access and all receipt identities.
No final pack or receipt means no claim of a complete mirror.

## Stop conditions and evidence limits

The owner confirmed on 2026-11-07 that no newer full-body CAD exists. Do not make
obtaining one a prerequisite for source-level composition. Retain the existing
pelvis/waist/legs and shoulder seats, reuse the corrected arm/gripper/head modules,
and distinguish a nominal derived head mount from a measured one. The source
shoulder spans differ by approximately 7.369 mm; an identity chest transform is
incorrect. Static table mounting, torso wiring and per-unit calibrations do not
automatically transfer to a legged assembly. Shared part corrections must propagate
through the modules, with tests of retained lower-body FK and waist-relative motion.

No qualified training claim with unverified attachment geometry, invented drive parameters, mismatched
embodiment identity or unverified retargeting. Every motion remains accounted for
even when infeasible. Preserve held-out splits and original/mirror families.
The current library still materializes all selected frames on the GPU; inventory
completion does not establish full-corpus training scalability. Policy exports
are not recovery checkpoints. Verify the authored trainer API is actually released
before reporting managed integration. No physical motion or run expected to exceed
the repository's approval threshold starts without the user's authorization.


## Managed retarget/cache/training candidate (2026-11-08)

The implementation is in `xmimic.managed_worker`, `retarget_workflow`,
`artifact_cache`, `checkpoint` and `managed_pipeline`. Run `tools/prepare_embodiment_job.py`
with a canonical body document or registered name, source BVH and explicit profiles;
`tools/managed_train.py` performs private publication and the CPU→GPU Job handoff.
See [managed training](../../../docs/managed-training.md) for execution bounds and
[runtime images](../../../docs/runtime-images.md) for the currently unqualified build boundary.

The owning nominal exporter commit is `96aabf07e16427d2c8e0006984cdd8d6d7e2d7cc`
(parent93141cd), separate from the corrected-body composition proposal. Its clean
archive decoded the portable39-coordinate development body and prepared the same
URDF/profile/target hashes in1.89seconds. Canonical body identity is not a registered
alias update, active station assignment or hardware calibration.

Observed real-source evidence: the first599-frame SOMA clip took86.45seconds on CPU.
At fixed2mm/0.01rad residual limits,423frames passed and176failed; maximum wrist
position residual23.863mm. The whole clip was rejected, with all599candidateframes
and residuals retained. No motion.npz or training candidate was minted for it.
The first8frames were feasible in0.343seconds; that bounded experiment did not admit
the complete source clip. Never silently crop, widen thresholds after seeing a result,
or report self-authored solver residuals as contact/balance qualification.

A synthetic whole-clip test runs the actual `TrainingContext`, canonical owner export,
CPU solver and immutable cache. Independent donated-JAX continuation restores replay,
optimizer, RNG, normalizers and randomized simulator constants exactly. Final complete
recovery is chunked for Catalog's1GiB/file limit; it still requires binding/template
validation when restoring. These tests do not establish actual Warp/GPU continuation.

```bash
# Use the exact owner checkout's generated proto and portable runtime without
# rewriting another session's shared editable installation.
PYTHONPATH=".:$OWNER/packages/sx-training:$OWNER/packages/sx-protocol/generated/python:$EMBODIMENTS" \
JAX_PLATFORMS=cpu python -m pytest tests/test_retarget_source.py \
  tests/test_retarget_target.py tests/test_retarget_workflow.py \
  tests/test_managed_worker.py tests/test_managed_pipeline.py \
  tests/test_artifact_cache.py tests/test_checkpoint.py tests/test_embodiment_bundle.py -q
```

| Failure | Cause | Correction |
|---|---|---|
| SOMA root/limb translations doubled | Declared translation channels replace BVH OFFSET | Honor channels on every joint; preserve virtual root and intrinsicEulerorder |
| Human hand target lies below the shorter robot arm | One root-relative scalar does not preserve limb proportions | Normalize declared source segments to owner-FK limb lengths; retain explicitanchors/holds |
| MuJoCo rejects full inertia with orientation | URDF tensor and inertial frame were both transferred | Rotate tensor into linkframe before writing fullinertia |
| Saved MJX fingerprint changes across processes | Static pytree repr embeds processaddresses | Fingerprint static array contents/types and scalar metadata |
| Rejected retarget loses paid coordinates | Cache forbade all unsuccessful artifacts | Retain diagnostics with training_admitted=false and no motion.npz |
| Reusing a completed cache reruns paid work | Caller ignored retained stage/operation identities | Verify content and reattach exact job; ambiguous acknowledgement refusesfreshwork |
| SDK wheel omits new runtime's dependency | Builder scanned packages/ only, Waddle lives underautonomy/ | Resolve the declaredworkspace closure through owning SDKbuilder |

Preconditions: approved source access, privateCatalogsourcecustody, explicitnominal
simulation approval, exactbodyassets and a feasible whole-clip mapping. Stop before
GPU learning on rejection, missingancestry, changedstatebinding, absentpublishedimage
or unavailableauthoringcontract. Keep licensing, original/mirrorfamilies and heldout
membership. Nothing in this procedure commands physicalrobotmotion.


Independent refutation caught the worker's wrong outputdirectory and interrupted
materialization poisoning an otherwise intact checkpoint. The corrected worker returns
`context.run_dir / "artifact"`; the test runs `sx_training.execute(context)` twice,
checks the owner receipt and forbids a second solver call. Native tars, copiedfiles,
JSON and manifests are admitted only after complete staged writes, with staging outside
the published artifactclosure. Mid-copy/mid-tar/mid-JSON failure tests refusepartial
admission and succeed onretry. The image provenance now reads the superproject's
mode160000 gitlink; `git rev-parse` inside an archive-filled submodule reports the
parent HEAD and is not embodiment evidence.


Final CPU suite:178 tests and44subtests passed in10.82seconds with
`JAX_PLATFORMS=cpu python -m pytest tests --ignore=tests/test_fast_equiv.py -q`.
The excluded upstream equivalence module is its own CLI harness requiring LAFAN/G1
motiondata and an explicitly constructed Env; it is not a pytestfixture suite and was
not exercised here. The independent correction pass passed20checks and retained no
finding. Hosted eviction/GPU restoration is still unverified.

Actual privateCatalog/R2 cache publication and complete sample readback succeeded for
the rejected599-frame retarget. ManifestSHA256:
`3553e5598010ed4074623edbe87a45d39057509fbac0956dc0b9b272f9c79df6`.
Pack: `xmimic.retarget.29a32a1dc5216ae845afb8cd516b1e5ac1b2d835c232912fbfcbb32f6c67401b`.
The three members (cachemanifest, candidatecoordinates and full residualprovenance)
were registered through Catalog, linked to the mirrored sourcepack, downloaded anew,
and verified. `training_admitted=false` remains intact. Private readback and operation
receipt are under `data/bones_seed/retarget-cache-readback` and
`data/bones_seed/retarget-cache-publication.json`. This proves retained cachecustody,
not successful retarget admission or a trained model.


### CPU retarget frame recovery (2026-11-08)

Use the exact owner runtime/embodiment checkout first on `PYTHONPATH`, without
syncing the shared venv. The bounded acceptance command was:

```bash
PYTHONPATH=".:$OWNER/packages/sx-embodiments:$OWNER/packages/sx-training:$OWNER/packages/sx-protocol/generated/python" \
python -m pytest tests/test_retarget_progress.py tests/test_managed_worker.py \
  tests/test_retarget_workflow.py tests/test_retarget_target.py tests/test_artifact_cache.py -q
```

Observed: 72 passed in 3.89 seconds. Real subprocesses exited after the third fsynced
frame of both a feasible and a rejected clip. Resume forbade solver evaluation of the
three committed frames and matched uninterrupted coordinates and diagnostics exactly.
The actual `sx_training.execute` path also stopped, checkpointed and resumed; missing
acknowledged frames refused recovery, and native progress ahead of the envelope survived.

| Failure | Cause | Fix |
|---|---|---|
| An interrupted whole-clip job retained only running status | No frame boundary was durable | Fsync a hash-chained frame journal before advancing; resume the exact warm start |
| Rejected frames disappeared from progress | Only successful outputs were treated as useful | Retain each frame's coordinates, residuals, iterations and disposition |
| Replacement could silently roll back beyond the managed checkpoint | Receipt and native cache were not compared | Bind receipt to the prefix digest and refuse missing acknowledged native state |

Preconditions: same admitted source/body/recipe/code, same execution reference, and the
same retained run volume. Stop on a changed binding, corrupt chain, missing acknowledged
frame or legacy running state without a journal. A kill inside a frame can repeat only
that uncommitted frame. These tests do not establish hosted Modal commit visibility.


The independent CPU review found that a missing final frame could look like a shorter
valid prefix when progress was inferred only from filenames. The corrected journal
fsyncs a separate count/hash tip after every frame, before acknowledging it. Restart
refuses a tip whose tail is missing; a complete frame ahead of an interrupted tip write
is verified and adopted without solving it again. The exact negative test kills the
actual managed runtime after two committed frames while its platform checkpoint is
still step zero, removes the last frame, and observes `ResumeRefusedError`. Together
with the prior focused suites, 74 tests passed in 7.79 seconds. Hosted Modal volume
commit visibility remains unverified; native fsync is not a provider commit receipt.

### Body review takes precedence over numerical acceptance (2026-11-08)

The owner rejected the rendered 39-coordinate composite as visibly broken and
requested human engineering review. The SO-101/ZED head must be replaced by the
original humanoid head on its original CAD neck seat. Keep training on that body
on hold. A nominal geometry contract, floor-only contacts, slower reference clock,
or a passing solver cannot establish that the assembly represents the robot.
Prepare front/side/three-quarter mesh views and the exact source/derived URDF for
the engineer; do not advance the body's registry alias or run GPU learning merely
because the generic workflow tests pass. The original SO-101 candidate, residuals
and source motion remain retained as diagnostic evidence.

The managed CLI now resolves registered names or portable body documents, verifies
all retained prepared bytes before reusing them, and shares one SDK-resolved endpoint
set between Catalog and training. `--profile preview` honors `SX_PREVIEW_ORIGIN` for
both. A reviewed tampering probe initially passed altered source bytes; a per-file
closure journal now refuses source, profile and scene changes before paid admission.
Correction verification passed six focused tests. The whole CPU suite before the
subsequent quality-projection changes passed 191 tests and 44 subtests. GPU metrics
are forwarded to nonblocking Waddle logging and retained as final CSV/JSON artifacts;
JAX, Warp and CUDA caches use the managed job's persistent cache directory. These
are implementation facts; no hosted GPU training or successful policy is established.

### URDF-to-MuJoCo rotation correction and original-head review

The first exporter used `eulerseq="xyz"`, which is intrinsic in MuJoCo, while URDF
RPY is extrinsic. At mixed angles (0.2,0.3,0.4), the rotation-matrix error reached
0.1195965; `XYZ` matched the embodiment FK to numerical precision. This corrupted
link and mesh placement, including a protruding battery that belonged inside the
chest. The exporter must use radians and uppercase `XYZ`. Consumer admission now
requires owner receipt v2 and `urdf-rpy-extrinsic-XYZ`, verifies the scene compiler,
and rejects older nominal bundles. Regenerate from exact canonical body + profile;
retain old receipts/caches as diagnostics, never silently relabel their physics.
Earlier nominal floor-clearance and collision conclusions using that exporter are
invalid. Pure source/URDF IK evidence is separate. No GPU run used those artifacts.

The original head restore is composition commit `5975a9de6390c64087d762ee0eea6344de8aa959`
in embodiments PR267. It uses `HEAD_MODULE` and `UPPER_BODY_MODULE.socket("neck")`,
removing the static body's SO101/ZED head and inferred shoulder-midpoint placement.
The resulting draft has33controlled coordinates; its two original head joints have
no invented drive binding. Nineteen head/assembly/asset checks passed. The actual
engineering renders instead evaluate owner `Kinematics` and convert each visual
origin to a quaternion, show only `<visual>` geometry, and label zero display poses.
The human review package is retained under `data/bones_seed/engineering-head-review`;
`candidate/view-relative.urdf` changes only mesh URIs, with joint/inertial XML checked
unchanged. It is a review draft, not physical or training approval.

### Check physical mesh continuity across composition cuts

The owner is the engineering reviewer and requires the whole model to be connected
and visually sensible. A single URDF tree did not meet that requirement: the restored
draft had 57 links and 56 joints, yet the shoulder meshes were visibly detached.
Compare the original full body, corrected static upper body and composite using the
same owner FK and literal visual geometry before changing a joint frame.

The source comparison found the fixed shoulder motors, stops and holders on the
corrected static body's `chest_link` (visual indices 5–12), outside the arm subtree
cut. The old full body had motor geometry inside the discarded shoulder mesh. The
composite therefore lost the fixed mounts when it replaced those arms. Sampled nearest
visual-vertex distances were 43.818/48.516 mm in the composite, versus 0.350 mm in
the original and approximately zero in the corrected static source. These are
diagnostic distances, not a calibrated contact or interference test. Both source
pitch axes already carry the approximately 15-degree tilt; adding another root
rotation would be an unsupported repair.

Restore source-owned fixed parts through shared embodiment composition, retaining
their source frames and explicit estimate provenance. Do not draw cosmetic bridges,
move joints to close a picture, duplicate torso inertia or infer measured per-part
mass from a single source hull estimate. Re-render the resulting candidate, compare
its source transforms and mass accounting, and obtain the owner's review before
admitting that body to GPU training.

The repaired review candidate is body
`ca3fb7c3b69a4df643d21904d64b57c7d954c536dbb70660143d75133e53f317`.
From the owning embodiment checkout, reproduce the visual review with:

```bash
PYTHONPATH=. python tools/review_humanoid_connections.py \
  --output /new/review-directory --previous-urdf /retained/previous/robot.urdf
```

The tool writes front/side/three-quarter, shoulder and ankle source comparisons
and a portable body/asset package. The observed packet is
`data/bones_seed/engineering-head-review/shoulder-restoration-v4`; its 59 links
form one connected tree. The fixed mount-to-chest sampled distances are now
0.0611/0.0242 mm without moving joints. Ankle rod and foot-seat geometry matches
the original and appears attached in closeups; the approximately 17 mm distance
between parallel linkage visual bodies was not an entirely open ankle.

Thirty-six affected checks passed. Independent review reproduced all 13 original
chest visuals exactly once, source transforms within 2e-8 and unchanged retained
FK at five nonzero configurations. Splitting the old monolithic chest hull requires
re-estimating the remaining chest and both mounts at the same 600 kg/m3 density:
their combined nominal mass changes from 4.79762 to 5.84242 kg. The full humanoid
adds 2.06566 kg of nominal mount estimates while retaining its original core
inertials. This is explicit estimate provenance, not measured robot mass. The owner
has been shown the revised packet; review and training approval remain separate
from these source-preservation checks.

### Resume publication after an external CI failure

Platform PR3562 retained exact head `f7ac42707a38b9508da38266e71c47702badcb6c`.
The image renderer check and 72 focused tests passed. Required run 42468120394 and
generated run 42468124053 stopped on HTTP 504 while downloading protocol tooling,
before the relevant checks ran. This is not a successful gate, published image or
hosted run. Do not repeatedly dispatch the same checks without evidence that the
download failure has changed.

Immediately after a push, `just land` read an old PR head while its dispatcher read
the new remote branch head; it then watched the old failed run. Compare PR head,
remote head and the existing run's `head_sha`. Reattach the already-dispatched exact
run and resume only its merge step after the required verdict; do not buy another CI
run to correct stale metadata. The temporary owner worktree was snapshotted and
removed with `tools/safe_worktree_cleanup.py`; pushed source and persistent review
artifacts remain available.

### Approved body: actual nominal simulation and reference evidence

On 2026-11-08 the owner answered **Proceed with nominal simulation testing** for
body `ca3fb7c3b69a4df643d21904d64b57c7d954c536dbb70660143d75133e53f317`.
That supersedes the visual-review hold for this exact body, not hardware qualification.
`examples/sentient_nominal.py` now binds it and explicitly holds the original neck
yaw, head pitch and waist rod at zero. The example's profile, recipe and training
metadata reproduce the prepared input bytes exactly.

Use the coherent owning snapshot `a5528ec576615cde3e0c6645b2a4fd548b72e942`, retained
on `Sentient-X/embodiments:codex/approved-nominal-export`. It combines reviewed
assembly a01160b with corrected exporter5780529. The older exporter's asset manifest
does not know the restored shoulder hulls. Do not treat a Python module overlay as
a published runtime or change the approved body to satisfy an old manifest.

```bash
# With the exact owner installed and SX_EMBODIMENTS_ASSETS naming the verified closure:
python -m examples.sentient_nominal --embodiment-document BODY.json \
  --source SOURCE.bvh --output NEW_PROFILE
python -m tools.diagnose_nominal --robot PREPARED_INPUT/robot \
  --output NEW_PD_DIAGNOSTIC --seconds 2
python -m tools.render_retarget --source SOURCE.bvh \
  --recipe PREPARED_INPUT/recipe.json --robot PREPARED_INPUT/robot \
  --cache CACHE/EXACT_KEY --output NEW_REFERENCE_VIDEO
JAX_PLATFORMS=cpu python -m pytest tests/test_render_retarget.py \
  tests/test_retarget_quality.py tests/test_embodiment_bundle.py -q
```

Observed: 24 focused tests pass. All 59 exported body frames matched owner FK at
nonzero native/floating-root pose (3.33e-16 m, 9.99e-16 rotation matrix error).
The two-second CPU baseline retained all 401 states, controls, forces, contacts and
warnings. It stayed finite with zero warnings but **fell forward**: root height
0.688 to 0.289 m, peak penetration28.2 mm. A finite simulation is not standing success.
The new tool renders the unchanged trajectory with an untrained-PD label; it refuses
existing output directories and paths under raw. It does not tune gains or command
hardware.

The shared `retarget_clip` function processed the complete599-frame SOMA clip in
88.76 seconds. The 2 mm optimizer goal still failed, while the predeclared30 mm,
0.01 rad, four-times-slower nominal contract accepted every frame. Observed maxima:
23.863 mm position residual,0.007526 rad orientation,21.983 mm root projection and
4.7612 rad/s joint speed; minimum collision geometry height2 mm. Retained source
timebase is120.004800192 Hz and nominal reference30.001200048 Hz. The source/reference
video is kinematic playback, separate from the actual falling physics baseline.

Cache key `7f2d793d4a9e66eb5171498c2c7d8698d14c686e88cdb87fe065ef46132d5233`, manifest
`04f2c340dc1402bb41b09c3904588988c7f77facc54e5754eb95498b6483a624`.
Reopening returned a verified cache hit in0.143 seconds without solving any frame.
`CatalogPublisher.publish`, `CatalogClient.record_source_lineage` and
`CatalogPublisher.restore` published the private cache, linked the mirrored source
pack and read back all four members with hashes verified. The cache remains
`training_admitted=false`; storage and nominal reference acceptance do not mint
Catalog training membership. Persistent receipts and both videos are under
`data/bones_seed/nominal-approved-ca3fb7/`.

| Failure | Established cause | Action |
| --- | --- | --- |
| Older exporter cannot resolve the new shoulder hulls | Owning manifest predates the approved assembly | Use the coherent source/manifest snapshot above |
| Stationary PD falls despite finite physics | Constant position targets provide no learned whole-body balance | Retain the fall and its trajectory; report no standing success |
| Retained archive probe contains G1 CSVs | That probe came from an earlier, different archive | Reuse the exact hash-bound SOMA prepared source; never retarget through G1 |
| Protocol generation remains unavailable | Exact protoc33.5 Linux asset still returns HTTP504 | Do not dispatch identical CI retries; no runtime image or hosted job is claimed |

Stop before hosted learning without published compatible images and Catalog training
admission. The finite physics result does not qualify balance, self collision, ground
contact parameters or hardware operation. Do not widen thresholds or crop failed
frames to turn a rejected reference into an accepted one.

The bounded independent review reproduced all 400 physics steps exactly and injected
a renderer failure: all 401 saved states and the diagnostic report survived. It found
no remaining issue in this change's input binding, passive holds or evidence claims.

### Use the existing URR boundary; test posture as well as endpoints

The owner rejected the earlier599-frame endpoint-only retarget despite small wrist
residuals. An independent anatomy audit found a0.38mm wrist error with77degrees of
forearm-direction error. The old recipe used downstream moving yaw links as fixed
anchors and omitted elbow/knee posture and chest motion. Never admit that cache from
its narrow nominal `accepted` field; the private review receipt is
`nominal-approved-ca3fb7/visual-fidelity-rejection.json`.

Use `sx_actions.urr.Motion` through `xmimic.retarget_motion`, preserving complete source
poses and explicit role correspondence, and `MotionBinding` for actual owner sites.
No robot proportions belong in the source Motion. Each timestamp rounds
`index * Fraction(frame_time_literal) * 1e9` independently; a repeatedly rounded
period accumulates clock error. Missing orientation must remain missing rather than
becoming identity. The retained shared source must have identical bytes/ID across
body or proportion changes. Private source artifacts do not belong in git.

Owner branch `codex/semantic-motion-sites` commit
`ec1e7fd8248af8ce6ce54ffb2a7ddbc29d9c4ced` adds shoulders through the corrected shared
arm modules and hips/knees/ankles through the leg modules. The humanoid has16sites;
new identity is `eb249c00dcff12dc212db1e6122df0ea4f13d57f17fc90d4c4a8855051e036f5`.
Removing only those8sites recovers exactca3fb7; URDF remains exacte7fd42 and all235
asset files verified. Do not rewrite previous identities, receipts or hardware acceptance.
Owner verification: `python -m pytest -q tests/test_sentient_pivot_sites.py
 tests/test_humanoid_correction.py` →10passed with exact owner and compatible shared
contracts/geometry packages. Exact owner snapshot and portable closure are retained in
`data/bones_seed/nominal-approved-ca3fb7/semantic-sites-v1/`.

```bash
# PYTHONPATH must select the exact owner snapshot; do not sync the shared venv.
JAX_PLATFORMS=cpu python -m pytest -q tests/test_retarget_motion.py \
  tests/test_retarget_target.py tests/test_retarget_workflow.py \
  tests/test_retarget_progress.py tests/test_retarget_quality.py \
  tests/test_render_retarget.py tests/test_artifact_cache.py
python -m examples.sentient_nominal --source SOURCE.bvh \
  --embodiment-document semantic-sites-v1/embodiment.json --output NEW_PROFILE
```

Observed focused suite:102passed,5skipped. The skips concern the installed
`sx_training` lacking `artifacts`; they are not hosted recovery proof. Real local
process-kill tests did verify directional-frame resume with no committed frame solved
twice. V1 checkpoints retain their bytes and cannot be reused for directional objectives.

| Failure | Established cause | Fix / evidence limit |
| --- | --- | --- |
| Accurate wrists, visibly wrong arms | Endpoints do not determine elbow configuration | Fit owner-defined limb directions and retain angular residuals |
| Missing generic anatomy | Approved body lacked shoulder/leg pivot sites | Add facts in shared owning modules; never infer sites from joint order |
| Initial feet fight leg directions | Nonneutral source feet calibrated to neutral robot feet | Fit8directions+chest first, then bind hand/foot rotations to fitted world frames |
| Static calibration stops before convergence | First frame hit60iterations; identical second frame converged after20more | Bound calibration at120iterations, keep the same5degree/.01rad limits; retain the failed attempt |
| Direction-only nominal projection crashes | Reduction over empty landmark arrays | Represent absent maxima as null; still enforce the direction gate |
| Direction result turns green from zero endpoint error | Unconstrained positions were interpreted as objectives | Mask them; separately display angular geometry and nominal quality |

Local DLS can remain in an incorrect bend basin with weak direction weights; a planar
ambiguity test needs explicit direction_weight1.0 while the generic default remains0.2.
Do not silently widen acceptance tolerances. Entire-clip failures stay retained and
unadmitted. First-pose calibration is not source cropping, flat-sole evidence, full-clip
success, policy training or physical qualification. Renderer SITE endpoints include
owner-defined local offsets transformed through actual MuJoCo link frames.

Complete corrected-calibration run `urr-profile-v3` processed all599frames in193.14s:
7frames meet geometric goals, maximum direction63.3445degrees, maximum orientation
52.8958degrees. Direction, orientation and speed checks fail; the candidate is rejected.
Median direction residual over all segments/frames is3.8571degrees, but that aggregate
must not hide large forearm failures. Cacheff52de1370648023536bbdd239537c270a684af55a0ebc964536dbd4e150c15b,
manifestb8d06cfdf17c1093ae01cd983caac01d9a191136cf3f14190f9d932d60935f21.
Both calibration variants retain identical shared Motion67eae1b41100387589ecb5e24a9d9f4513639c1581637b2e715bbb3c861f45cd.
The earlier neutral-foot variant passed0frames; its599frames remain separately retained.
This run establishes shared representation/custody and honest rejection, not solved
retargeting or training readiness. Next investigation is the constrained solver's remaining
forearm fit, including limit saturation and orientation tradeoffs, without altering raw labels.

### Generality audit: remove application-specific heuristics

The owner explicitly requires generic algorithms; embodiment-name branches and special
adjustments signal missing abstractions, even inside examples or JSON. First classify
physical facts/source semantics/explicit experiment assumptions separately from inferred
behavior. Preserve input declarations; fix the owning capability for the latter.

Observed fixes: removed knee/hip substring seed offsets, finger-name hold selection,
terminal wrist/foot name holds, Foot-suffix scale selection and mesh-only support math.
Shared actions commit7507a26c (`sx` PR3579, stacked on PR3562) supplies
`MotionBinding.joint_names` and `correspondence_scale`. Bound-site ancestry includes
shared ancestors and off-path mimic leaders but excludes unrelated components. Hold its
complement at the supplied reference state; never freeze coordinates because their
initial Jacobian happens to be zero. Explicit corresponding polyline roles determine
scale through one shared operation, invariant to independent rigid placement and without
mutating source motion. The example consumes these capabilities instead of hiding the
old conditions in a per-robot configuration.

Single-sample geometric fitting now accepts the real first frame; no duplicate synthetic
frame is needed. Source/training workflow and NPZ export still refuse a one-frame training
motion. With exact source/owner/URR packages, the actual Sentient reference calibration
passed at unchanged tolerances in42iterations, worst direction4.8262degrees, without
named posture tweaks. Evidence is `nominal-approved-ca3fb7/generic-reference-v1/`.
This is one calibration, not another whole-clip or trained-policy success.

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests/test_retarget_generality.py \
  tests/test_retarget_target.py tests/test_retarget_workflow.py \
  tests/test_retarget_progress.py tests/test_retarget_quality.py \
  tests/test_retarget_motion.py tests/test_artifact_cache.py tests/test_render_retarget.py
```

Observed106passed,5existing skipped managed-runtime tests (installed package lacks
`sx_training.artifacts`). The new generality tests use actual owner bodies with one and
three non-anatomically named branches and an unobserved branch. Renaming all joint/link/
site/source-role/component identities and permuting native/URDF order preserves the
physical solution to1e-10. Shared actions35tests passed; strict Pyright and Ruff passed.
Its exact35-file source snapshot and commit/hash receipt live under
`generic-reference-v1/dependencies`; do not repoint the shared venv. Isolated worktree
was removed after verification, and no redundant CI or paid run was dispatched.

| Failure | Cause | Owner/fix |
| --- | --- | --- |
| Example behavior depends on anatomical substrings | Missing dependency and reference-state abstraction | Shared MotionBinding ancestry plus supplied reference state |
| Calibration invents a second frame | Tracker duration leaked into geometric fitting | Permit one geometric sample; preserve training-duration admission |
| Generic claim despite duplicate coupled solvers | Shared action solver also requires executable action semantics | Separate shared geometric problem/result from action admission, then remove duplicate consumers' implementations |

Do not claim the last owning-layer gap is resolved by these tests. Numeric generality,
shared ownership, full motion fidelity and physical qualification are separate evidence.

### Preserve absolute foot direction with physical point pairs

The 2026-11-08 foot review found that first-pose-fitted foot orientation offsets
could hide a wrong heading. The historical `urr-profile-v3` frame-zero orientation
residual was near zero while physical foot-forward versus observed ankle-to-ball
angles were110.50degrees left and98.26degrees right. Those different vectors are
not the metric to use for a corrected ankle-to-toe comparison.

Inspect literal owner CAD before changing frames. Both feet point toward local+X;
there is no supporting evidence for a per-foot90/180degree correction. Source
ankle-to-ball slopes downward, so treating it as horizontal sole-forward creates
another mismatch. The owner added fixed `left_toe_edge`/`right_toe_edge` sites at
literal40mm front sole-edge midpoints (zero distance to the edge) in shared leg
modules, commit`aa3c97f421a961ff4ede1cb5518ece7e25efba78` (13tests pass).
Body`1fb0d53b81fca5fcd8c09b00e796a31e08313fa2ea8bc22b4248dcb8b29eda3c`
retains approved URDF`e7fd42a06794c5d3fce549e55630ea0a9194eee04411b7f06a9d7ed5ada87e67`;
removing only the two new sites restores`eb249c`. No geometry, joint, limit or
hardware acceptance changed.

Use existing generic `Direction`: observed `Foot→ToeEnd` corresponds to physical
ankle→toe-edge. Exclude feet from fitted orientation offsets. The obsolete
`SiteAxis` prototype is retained in experiment history only; shared forward-revert
`c3b1d5b63a8acf8a02b9bc478634f01d195518b8` restores the7507actions source tree.
One point-pair direction does not determine full sole orientation, contact or slip.

The viewer's `raw_direction_metrics` independently compares actual MuJoCo FK
point-pairs to unchanged source pairs for every source frame, including frames
omitted from display sampling. Gravity-plane heading is null when undefined.
For before/after evidence use the same physical endpoints and observed source pair
on both trajectories. A retrospective CAD-site measurement on an older body must
record its original candidate identity separately from the reference semantic body
and verify identical URDF geometry; never relabel or rewrite an old artifact.

```bash
# Exact toe-sites-v1/owner and compatible shared sources first on PYTHONPATH;
# SX_EMBODIMENTS_ASSETS selects its verified asset closure. Do not sync the venv.
python -m examples.sentient_nominal --source SOURCE.bvh \
  --embodiment-document toe-sites-v1/embodiment.json --output NEW_PROFILE
python -m pytest -q tests/test_retarget_direction_frames.py \
  tests/test_retarget_generality.py tests/test_render_retarget.py
```

Observed at this stage:24tests pass. The first new calibration stopped after4steps
with large residuals. Finite-difference diagnosis found a nonzero projected gradient
(norm0.0988573): clipped unconstrained damped-least-squares steps all increased
residual, while a feasible projected-gradient step reduced norm0.32032439→0.29104.
Retain this rejected diagnostic; do not adjust named joint seeds or tolerance to
hide an optimizer failure. Source vectors remain immutable, limits unchanged, and
only complete accepted motion may proceed toward training admission.


The projected-gradient correction now has a reachable planar regression:
`python -m pytest -q tests/test_retarget_bounds.py`. Old code stopped immediately
at `[2,-2,.3]` with2.666m error; the generic correction reaches independently known
`[1.1,.6,.3]` within unchanged bounds/holds/tolerances. It fixes false stationarity,
not global feasibility. No SciPy runtime dependency was added.

New `toe-retarget-v2` calibration remains rejected after120iterations: toe errors
1.54/1.66degrees, leg errors9.56–11.30degrees. The identical point-pair metric on
old frame zero was107.36/94.05degrees; its full599-frame means were98.62/92.12degrees.
Private `toe-comparison-v1/previous-candidate.json` binds both semantic body IDs,
the common URDF and all599measurements. Do not call this a new full-clip success.
Independent bounded least-squares diagnosis validates the analytic Jacobian to
6.42e-11; two starts reach the same local leg/toe compromise, with both terminal
ankle coordinates at±25degrees. One start converges in14evaluations with arms and
chest nearzero, legs9.66–11.24degrees, toes1.26/1.34degrees. This separates slow
optimization from a remaining local kinematic-fit limitation; it does not prove
global infeasibility. The diagnostic, retained-state source hashes and script live
under `foot-direction-audit/bounded-least-squares-diagnosis.json` and
`bounded_least_squares_diagnosis.py`.

Aggregate affected check:91passed,5skipped,32subtests across retarget source, Motion,
target, generality, physical direction frames, bounds, workflow, progress, quality
and viewer tests. Skips reflect the installed `sx_training.artifacts` dependency
being absent; they are not hosted-training verification. No new full-clip motion,
GPU execution or training admission was produced from rejected calibration.

| Failure | Cause established by this run | Correction/boundary |
| --- | --- | --- |
| Small fitted foot-orientation error but wrong visible heading | Arbitrary first-pose offset redefined desired heading | Map observed source point-pair to physical canonical point-pair; independently audit raw directions |
| New solve stopped after4iterations | Clipped unconstrained step was not a descent step despite nonzero projected gradient | Generic feasible gradient fallback; reachable independent regression |
| Better toe direction with10–11degree leg errors | Two bounded local fits reach the same compromise with terminal ankle limits active | Preserve rejection and model-limit evidence; do not fabricate a training recipe, widen bounds or crop frames |


The follow-up owner FK audit (`foot-frame-audit/leg-axis-audit.json`) found original
and current leg axes and zero-pose FK identical. Knee/foot1 axes are perpendicular
to physical foot-forward; foot2 is parallel, confirming terminal roll rather than
a sideways knee. At the bounded optimum, native knees are−3.16/+3.97degrees while
hip→knee→ankle point-pair bends are23.58/22.76degrees. Fixed link-origin offsets make
those different quantities. Do not mistake physical pivot polylines for anatomical
bone angles, infer a joint-axis repair from the residual, or claim global
infeasibility. Correspondence semantics and geometric feasibility remain to resolve.
The inspected source/calibration foot-detail visual and hash receipt are retained in
`toe-comparison-v1/visualization-v2/`; all panels label both fitted poses REJECTED.

### Assess PolicyIK plus model-based URR tracking before training

Research inspection on2026-11-08, not an implemented or qualified recipe:
- PolicyIK paper: https://arxiv.org/html/2611.05372v1 (Sections4,5.2, AppendixJ).
- Released implementation: https://github.com/Tinaliu0123/PolicyIK
  `simulation/policyik_sim/tasks/actions.py` and `env_cfg.py`.
- SuperMimic method and benchmark limits: https://supermimic.github.io/.

PolicyIK accumulates joint references, observes reference-minus-measured joints,
uses coarse/fine pose error and error-dependent action scale, and rewards progress
and holding. Simulation uses PPO; physical OpenArm uses FlashSAC/HER. AppendixJ
explicitly trains separate xArm7 policies: evidence for a reusable formulation,
not one checkpoint transferring unchanged across morphologies. Hardware pose
errors use encoder FK, not independent external metrology. The98.9percent simulated
figure includes a deployment protocol;93.5percent is single-attempt. Neither is a
full-body walking or arbitrary-embodiment guarantee.

Read local boundaries before describing this as a configuration change:
```bash
rg -n 'floating root|floating-root' xmimic/model.py
rg -n 'GOAL_Q_OFFSET|self.q\[fut\]|root_p|joint_names' xmimic/motion.py
rg -n 'def _anchored_ref|def reward|def termination' xmimic/mdp.py
sed -n '1,135p' ../glued/packages/sx-actions/sx_actions/urr.py
```
Observed: model loader requires one floating root; motion goals carry native q/qd;
tracking rewards use yaw/translation anchoring and root height/tilt termination.
URR Motion already owns sparse named Position/Pose samples, frames and time, with
missing observations preserved. Thus direct URR fixed/mobile tracking needs an
explicit task/state/controller boundary refactor, not a pretend floating joint or
synthetic missing-foot targets.

Proposed experiment: same URR goal contract, per-embodiment trained controllers,
robot-owned mounts/base mobility/limits/sites; compare bounded numerical IK,
PolicyIK-style model-free RL and the same task trained with XMimic's learned
world model. Sample reachable configurations and compute canonical FK goals before
adding UMI or BONES-SEED trajectory curricula. Observed source poses remain immutable;
role/site selection and feasible task priorities are separate from source identity.
Missing orientation is not zero, one direction is not full foot orientation, and
human pivot geometry is not automatically a homologous robot bone frame.

A learned dynamics model must condition on accumulated reference as well as measured
state; otherwise identical state/action can produce different transitions. Retain
analytic canonical FK for pose rewards rather than relearning known kinematics.
Adapt static-goal progress/holding rewards for moving references, and ensure any
near-goal action scaling cannot suppress unrelated balance corrections. Use smooth
precision losses for imagined gradient updates and exact tolerance/contact checks
for evaluation. These are design hypotheses, not adopted implementation choices.

Stop short of a universal-checkpoint or millimetre physical claim without held-out
embodiment tests, tracking/holding errors, delay/disturbance/contact/constraint
measurements and external pose metrology. A world model's broad motion reward or
low average prediction loss does not establish sub-millimetre tracking fidelity.
The current foot-calibration failures remain rejected; learning must not erase them.

### Preserve baseline behavior when combining training recipes

The owner made comparative experiments an explicit requirement on2026-11-08:
vanilla XMimic, vanilla PolicyIK and the combined recipe must remain separately
measurable. The intended output is precise URR execution of UMI, teleoperation and
BONES-SEED trajectories, including gripper/hand pose and measured gripper aperture.
Read [the benchmark plan](../../../docs/urr-controller-benchmarks.md) before changing
the controller/training boundary. It is a specification; its evaluator and paired
training runs are not yet implemented or executed.

Native reproduction and matched-task adaptation answer different questions. A
PolicyIK-derived controller ported to another simulator/body is not vanilla PolicyIK;
an XMimic-derived fixed-base URR controller is not vanilla XMimic. Retain upstream
protocols first, then label and control adaptations. Preserve seed/task/split pairing,
predeclare metrics and practical non-inferiority margins, and report inconclusive
intervals explicitly. Never infer no regression from higher aggregate reward or
one favorable seed. Evaluate each required capability and absolute task acceptance.

Useful current boundary checks:
```bash
rg -n 'def observed_frame|apertures|def select' ../glued/packages/sx-actions/sx_actions/urr.py
rg -n 'floating root|floating-root' xmimic/model.py
rg -n 'GOAL_Q_OFFSET|self.q\[fut\]' xmimic/motion.py
git diff --check
```
Observed: shared URR already carries metric apertures and explicitly framed pose
observations through `observed_frame`. This is not an episode importer or closed-loop
aperture-policy qualification. Current XMimic still uses a floating-root/native-joint
reference boundary. Keep missing aperture distinct from zero and teleoperator commands
distinct from measured motion; joint-only source episodes need their own canonical FK.

| Failure/signature | Cause or uncertainty | Required evidence/correction |
| --- | --- | --- |
| Combined recipe appears better because original methods were weakened | Adapted methods mislabeled as native or unequal tasks/budgets | Native reproduction plus labelled matched comparisons and component experiments |
| Visually improved human retarget still reports10–11degree leg errors | Anatomical/pivot correspondence includes fixed offsets; not measured control error | Separate known-feasible robot tracking from source correspondence, then measure full physical task constraints |
| Accurate hand target while gripper response/balance regresses | Aggregated reward hides coupled objectives | Per-role pose/aperture/contact/temporal metrics and all required promotion criteria |

The current failed human cache stays unadmitted. Known-feasible synthetic and measured
robot trajectories can establish baselines while correspondence is unresolved. No
paid training, new tolerance acceptance or physical motion was performed by writing
this plan. Stop short of claiming benchmark enforcement until the evaluator executes
the frozen protocol and retains its complete results.

### Execute stationary pose/aperture comparisons and native controls

The owner explicitly requested the runner and real baseline execution on2026-11-08.
Use [stationary results/commands](../../../docs/stationary-benchmark-results.md),
[the experimental contract](../../../docs/urr-controller-benchmarks.md) and
[native protocols](../../../docs/native-baselines.md). CPU adaptations and native
reproductions are separately labelled. A completed pilot is not converged training.

Owner83f32c4 adds fixed-base, prismatic and explicit affine-dependent projection to
the existing exporter. Dual PiPER fixture0c19cd156a9e6010a40cc5d6ba24755351ac43cb6893c785bf80d5a3b097df48
uses12arm coordinates and2metric gripper drives, plus2source mimic followers.
Hand sites/aperture law come from canonical parts; Y+/-0.35m mounting and PD settings
are explicit nominal assumptions. Sentient aperture is undocumented; ALOHA's native
slide range/mimic disagrees with its URDF. Exclude those unsupported aperture cases
rather than inventing facts. Conservative PiPER native limit rounding is a subset
of source bounds; export the native subset, refuse extensions. Owner15tests,
strictPyright0errors and the real all-link FK/aperture test pass. Constant PD shows
16.029mm/3.338degree gravity sag; exact geometric FK still agrees to1.6e-16m.

First complete scalar screening:13cells,3seeds for each learned method,20updates
of128real transitions per seed,8common evaluation goals,5s/goal and final1s continuous
holding. Both hands must satisfy1mm/0.2degree and both apertures1mm. DLS held8/8;
all learned adaptations0/24 each. PrecisionPPO had19/24 and combined14/24 recorded
constraint-failure tasks. Total166.78sCPU. This rejects promotion of that pilot;
it does not prove the research combination cannot converge. Retain old source,
checkpoints and measured native states before every correction. The measured replay
uses visual group1, not collision group0; targets display URR frames rather than
invented target joints. Aperture readings are owner-law-derived proprioceptive
values, not independent external jaw metrology or grasp-success evidence.

Whole-observation WM prediction gave a network authority over known goals/error
channels. Structured revision3f3e089 learns q/qd only, deriving commands, FK, aperture,
precision and history. Extend existing Kinematics (hinge/slide/affine-dependent,
anchors/reference offsets), never create another embodiment-specific FK. Legacy
metadata outputs remain bitwise equal. Actual PiPER reconstruction discrepancy
<=2.21e-6, aperture<=5.6e-9m, discarded known-channel gradients0. Existing rotation
log gained optional safe_zero handling; default vanilla arithmetic is unchanged.
Moving structured training refuses missing trajectory state; moving feedback eval
has separate timing criteria. Evaluation seeds0–7 are development evaluation;
10000–10063 are reserved for final confirmation before tuning.

Scalar collection548transitions/s was the next measured blocker. Revision7e1e34a
shares immutable MjModel with separate MjData/RNG/goals per lane and a bounded8-thread
physics pool; batched known observation/reward math yielded16,102transitions/s
(29.38x), warnings0. Complete10-update64x32 training measured3,507/sPPO and2,199/s
combinedWM, including JIT, optimizer updates and durable checkpoints. Collection-only
speed is not full training throughput. T-by-N GAE handles each lane's terminal before
flattening; replay/step counts includeN. Complete integration/RNG/command/goal/reset
state resumes exactly.27focused tests pass. `LearningConfig(num_envs=64)` is an
execution parameter; fixed bodies/simulator semantics and task thresholds stay equal.

```bash
python tools/benchmark_urr_vector.py --prepared PREPARED --output NEW.json --num-envs 64 --steps 128
python tools/benchmark_urr.py --prepared PREPARED --output NEW_RUN \
  --methods dls ppo_coarse ppo world_model combined --seeds 0 1 2 \
  --num-envs 64 --iterations 500 --rollout-steps 32 --episode-steps 250 --hold-steps 50
python -m pytest -q tests/test_kin_scalar.py tests/test_urr_vector.py \
  tests/test_urr_benchmark.py tests/test_urr_learning.py tests/test_stationary_benchmark.py
```

The native worker only runs pinned upstream source/tasks/checkpoints via managed
Jobs, preserving TRAINING versus EVALUATION(false) purpose. PolicyIK needs its native
Python3.11/IsaacSim5.1/IsaacLab2.3.2/RTX stack; H100/A100 lack required RT support.
XMimic native uses H100/CUDA12 and its40checksum-boundG1/LAFAN motions (75files,
110,153,716bytes), exclusively the native control, never Sentient retargeting.
Native acquisition/imports/unit tests are not native scores. Real worker validation
must call actual `TrainingContext.execute` and verify its owner receipt before a
paid launch; worker must return exactly run_dir/artifact. Publication uses existing
CatalogPublisher, private EVALUATION/false evidence, verified archive inventories
and readback; no parallel R2 or Modal launcher.

| Observed failure | Cause | Correction |
| --- | --- | --- |
| Returned pose lags current qpos by1.6–1.8mm | mj_step integrates after last position stage | mj_forward before measuring returned state; canonical FK consistency regression |
| Exact-target precision gradient NaN | Euclidean/SO3 norm undefined derivative at zero | Safe zero subgradient with exact norm values; partial/exact-zero tests |
| Copied task result silently reused | Task lacked its request/policy identity | Bind/verify per-task request hash, seed and sample count |
| Tiny unit mechanism autorestes numerically | High damping on tiny fixture inertia | Explicit fixture armature and no-warning assertions; no PiPER gain adjustment |
| Successful paid child rejected at framework boundary | Returned native-output instead of managed artifact root | Actual owner execute/receipt test plus atomic complete output publication |
| Cancelled child leaves grandchild | Waiting only for leader exit | Kill/reap entire process group, including SIGTERM-ignoring descendants |
| Extra source/output or symlink accepted | Incomplete inventories/path validation | Exact declared source/output closure and regular in-root paths |
| Publication cannot resume after fsync/link fault | Unidentified staging temp blocked strict inventory | Temp identity binds destination+digest; reconcile only known exact bytes; forbid paid child rerun in regression |

Preconditions: immutable body/runtime/task/source pins, true measured-transition
collection, disjoint train/development/final identities, bounded managed spend.
Stop on physics warnings, unsupported mobility/joints/aperture law or failed owner
admission. Do not tune thresholds after seeing scores, admit failed human retargets,
claim native scores from CPU adaptations, or count cached source acquisition as a
completed GPU baseline. Update actual native receipts separately when they exist.

### Audit reward fidelity before extending a matched cohort

On 2026-11-08 the `vector-million-v1` cohort was interrupted after a counterexample
in the adaptation's objective, not because upstream PolicyIK or XMimic had been
shown to fail. Preserve its source checkout `bcae2a6`, request and all numeric
checkpoints. Coverage: DLS8/8; coarse PPO, precision PPO and coarse WM each completed
three seeds at500updates and held0/24; combined seed0 completed and held3/8, seed1
stopped at checkpoint232, seed2 was not run. This is **not a complete cohort**.
Combined seed0 terminal medians0.425/0.352mm and0.153/0.111degrees are useful partial
evidence, but its worst second-hand position8.316mm and orientation1.917degrees
miss strict holding. Do not suppress that positive result or turn it into a
non-regression/promotion claim. Exact paths are in
`data/stationary-benchmark/vector-million-v1-interruption.json`.

The frozen scalar-coordinate Gaussian mean rewarded sacrificing one axis:
(20,20,0)mm→(35,0,0)mm increases Euclidean error28.284→35mm while precision reward
rises0.859759→0.928572 with other terms equal. It also depends on world-axis choice.
The TRAIN-only audit replays4096transitions from three seed0/update500 cells,
verifies checkpoint leaves and recomputes recorded reward within1.8e-7. All scalar
position errors>=5cm had objective derivatives<1e-6/m. Full native-q gradients were
not zero; these are not measured PPO optimizer gradients. The helper does not
step physics, optimize, or read evaluation transitions.

```bash
# Preconditions: persistent owner83f32c4 and exact compatible sx-actions,
# sx-geometry/sx-contracts on PYTHONPATH; the frozen checkout still retained.
python data/stationary-benchmark/objective-audit-v1/audit.py
python -m pytest -q tests/test_urr_reward.py tests/test_urr_benchmark.py \
  tests/test_urr_vector.py tests/test_urr_learning.py
```

The audit helper prepends the historical checkout and verifies its source hashes,
so newer core code cannot silently change the audit. `results.json`, `REPORT.md`
and `reproduction.log` record actual outputs and source links. Review actual pinned
PolicyIK22f3a5c `tasks/env_cfg.py`, `mdp.py`, `actions.py` plus inherited OpenArm
bad82e23 terms and IsaacLab2.3.2 dt integration; an observation scale is not a
reward width. Native XMimic8f7693f uses whole-body vector tracking, model rollout
losses, critic/bootstrap/termination and optimization machinery absent here.

Source-term correction `cd3eb346da11388ed26d5feaceacc9628fa7bd97` passed37focused
checks. It uses Euclidean position and three centred45cm virtual keypoints, literal
source weights/widths, transition-based precision shaping with first/changed-goal
reset, exact source gain ramps, native-normalized margin cost and control-dt scaling.
The aperture position pair and global multi-role ratio are documented extensions;
no source-native gripper reward is invented. Existing normalized action-rate cost
remains; do not copy radian velocity costs onto metre-valued slides. World-frame
observation features, native action scales/governor and small learner architecture
remain labelled adaptations. Neither numerical source agreement nor tests prove a
learning gain. The subsequent zero-aperture follow-up replaces scalar abs by the
safe scalar norm in reward/ratio and tests exact-zero gradients plus two-sided
descent through affine/curved owner laws. Follow-up455390a passed37focused tests.
TRAIN-only `reward-pilot-v3` completed all four adaptations at10updates×64×32,
20,480real transitions each, with finite recorded objectives/losses and no evaluation.
The separately authorized `vector-million-reward-v3` completed from frozen455390a
in1728.653s, same500update/three-seed budget and held thresholds. DLS8/8 constraint0;
coarsePPO0/24 constraint24; precisionPPO15/24 constraint0; coarseWM0/24 constraint0;
combined12/24 constraint0. Report `do_not_promote`: combined-minus-precisionPPO held
failure+.125, mean per-task terminal-p95 position left+.400mm/right−.323mm. These
mixed descriptive paired results do not establish non-regression, native baseline
performance, convergence or physical qualification. All104development tasks and
12final complete learned checkpoints remain recorded; final seeds stay reserved.
Replays `vector-million-reward-v3-precision-replay` (DLS/precisionPPO) and
`vector-million-reward-v3-comparison-replay` (precisionPPO/combined) each show only
training-seed0/task0. Never generalize that single video into the cohort result.

A new objective requires a new frozen code/request/run. Never change old evaluation
bytes, widen thresholds, tune weights on development scores while calling them
source-derived, resume old checkpoints under altered semantics, or infer native
failure from an adapter. A short TRAIN-only stability pilot precedes any newly
approved cohort; it is not permission to run another full cohort automatically.
Strict per-role continuous acceptance remains authoritative even when reward rises.

| Observed failure | Cause | Correction / stop condition |
| --- | --- | --- |
| Reward rises while Euclidean pose worsens | Mean of narrow scalar-axis kernels rewards sacrificing coordinates | Literal source vector/keypoint terms; common-frame invariance and analytic counterexample regression |
| Far positional error gets negligible direct reward slope |1cm observation scale was reused as tracking kernel width | Source negative distance and .20/.05m keypoint kernels; inspect TRAIN replay gradients without claiming PPO gradient collapse |
| Precision progress/dwell differs from source | Linear best-ratio improvement and growing dwell were substitutes | Exact utility/potential/backslide/binary dwell with explicit transition/reset state |
| Exact-goal aperture gradient pushes away from target | JAX scalar abs chooses a nonzero derivative at zero | Safe scalar norm with exact-zero and positive/negative error regressions |
| Single successful combined seed becomes a whole-cohort claim | Partial cohort coverage and worst-role errors hidden by medians | Retain every cell status, worst errors and unchanged held-pose rule; incomplete remains inconclusive |

### Preserve completed screening evidence through Catalog

Verified first screening pack:
`xmimic.benchmarks.stationary.screening-v1.ec27776bf3b8`, project
`b98ba48a-39c1-4346-98a4-d9c130144a42`, manifest
`69663086f02ba41de14c2b87e740c33bef7916a8f1dcc77e05454e8151898e47`.
Complete readback verified8objects/54,948,901bytes covering1,937logical files:
prepared body/assets, all task/training results, latest complete recovery checkpoint
per learned cell, whole historical checkout, owner source/licenses and measured v3
visuals. This receipt covers the **first screening only**, not the interrupted
million-transition cohort. All older local checkpoints remain intact.

`tools/publish_urr_benchmark.py` uses existing CatalogPublisher.publish_files and
restore_pack. The actual receipt is
`data/stationary-benchmark/screening-v1-catalog/publication-receipt.json`;
transport/publication.json retains the upload/registration operation identities.
Manifest purpose EVALUATION and training_eligible=false are mandatory; evidence
publication is not Catalog learning admission. No direct R2 SDK or new bucket.

Reproducing publication needs the merged sx_training ArtifactPurpose.EVALUATION
owner and existing approved private-project credentials. Never print credentials.
Use a fresh derived preparation directory for new evidence; attach the original
journal for the same immutable evidence. Stop on changed hashes, incomplete cells,
manifest-purpose mismatch, unknown service outcome or archive larger than1GiB;
shard explicitly before any new publication. Preserve licensed source and exact
logical archive inventories, including empty source files. Verify readback before
claiming durable evidence. This procedure does not authorize a new training run.

Publisher correction278b63f was independently checked against the two reproduced
preservation failures. Its13tests passed; separate probes rejected missing cells,
broken request/result links, changed assets and owner URDF before archiving,
preserved a competing destination byte-for-byte, and recovered exact staging bytes
from interruptions both before and after a successful hard-link seal. No network
publication was performed by this correction review. Evidence is retained under
`data/stationary-benchmark/publisher-review-v1/{probe.py,results.log,correction.py,correction.log}`.

```bash
# Requires the merged sx_training.artifacts owner on PYTHONPATH.
python -m pytest -q tests/test_publish_urr_benchmark.py
```

| Publisher failure | Proven cause | Corrected behavior |
| --- | --- | --- |
| Incomplete or changed screening accepted as complete | Trusted complete=true and supplied cells without validating linked receipts and prepared asset closure | Verify request/result/report hashes, exact requested cells, executed source hashes, prepared scene/owner/assets and evaluation task identities/counts before archiving |
| Concurrent destination replaced while sealing an archive | Existence check followed by overwriting rename | Atomic no-clobber hard link; verify a competing destination and preserve it on mismatch; reverify known complete staging bytes before recovery |

A corrupt or unknown staging file remains a refusal, not permission to overwrite
retained evidence. Archive verification and successful local correction do not
mint training admission or substitute for Catalog readback of a newly published pack.


Publisher inventory v2 removes historical-folder assumptions. The full checked
`checkout/` is canonical; `source/` is optional supplementary evidence. Supply
recorded replay directories explicitly (repeat --visual-review for multiple views):

```bash
python tools/publish_urr_benchmark.py --screening COMPLETED_RUN --prepared PREPARED \
  --owner EXACT_OWNER_ARCHIVE --visual-review PRECISION_REPLAY \
  --visual-review COMPARISON_REPLAY --output NEW_PUBLICATION_DIRECTORY \
  --project PROJECT --pack NEW_PRIVATE_EVIDENCE_PACK
```

Without --publish this only prepares local archives; it grants no network write or
training admission. The visual archive has canonical group `visual-review`, retains
each source directory's name and bytes, and rejects duplicate names or replay
body/scene/task receipts outside the completed run. Inventory records publisher
SHA-256; Catalog generator provenance carries that same identity. The initial v1
pack stays readable through its existing manifest/readback; do not rewrite it to
v2 or use its directory for new preparation.19focused publisher tests passed with
an exact archived owning `sx-training`/`sx-contracts` atf0607b1. The removed worktree's
PYTHONPATH failed because the installed SDK predates artifacts; use the retained
`data/native-baselines/runtime-owner-f0607b1/packages/` roots and source-receipt.json
instead of changing the shared environment. No new cohort archive or publication
was created during this implementation/test pass.

After bounded review of41d7022, actual local preparation of the completed corrected
cohort produced `data/stationary-benchmark/vector-million-reward-v3-catalog/`:
8objects/72,138,011bytes,6archives/1,960logical files,12complete final checkpoints
and10members from both replay directories. Manifest
`eb3187da1c57fd2c5e9052467f864bf82d3781fe241ceca61a1f2ba0e635f199` is EVALUATION/false.
Independent manifest/object/archive verification passed;9first-screening publication
files remained byte-identical. `vector-million-reward-v3-preparation.json` records
all hashes and totals. This stage established local preparation only; publication
and readback were separately executed afterwards.

The authenticated publisher then uploaded and fully restored all8objects through
Catalog, verifying all6archive inventories/1,960logical files. Private pack:
`xmimic.benchmarks.stationary.vector-million-reward-v3.455390a5`; manifest is the
same `eb3187da…635f199` above,72,138,011bytes. The final
`vector-million-reward-v3-catalog/publication-receipt.json` has
`verified_readback=true`, purpose EVALUATION and `training_eligible=false`.
Use the same command with `--publish` and existing project authentication; never
discard its durable `transport/publication.json` to retry accepted work. The slow
laptop readback completed without a second publication or alternate R2 transport.
Private storage of the do_not_promote result establishes evidence custody, not
policy acceptance, native upstream scores or permission to train on evaluation.
