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
