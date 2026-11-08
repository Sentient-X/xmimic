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
