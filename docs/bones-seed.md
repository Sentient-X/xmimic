# BONES-SEED source custody and inventory

The full source is [`bones-studio/seed`](https://huggingface.co/datasets/bones-studio/seed),
pinned to `2f59b2077b9da34dd4e43618e705c7cb962c9a66`. The preparer inventories
every motion in its actual v004 metadata. It downloads only the 4.5 MB parquet,
using an existing Hugging Face credential with dataset access; it never accepts
terms or downloads motion archives. Already downloaded metadata can be used offline.

```bash
uv run --no-project --with huggingface-hub --with pyarrow \
  python tools/prepare_bones_seed.py

# Offline, with exactly the pinned v004 metadata bytes:
uv run --no-project --with pyarrow python tools/prepare_bones_seed.py \
  --metadata /path/to/seed_metadata_v004.parquet

# Optional inventory of file presence after an independently authorized extraction:
uv run --no-project --with pyarrow python tools/prepare_bones_seed.py \
  --metadata /path/to/seed_metadata_v004.parquet --local-root /path/to/bones-seed
```

The default output is `data/bones_seed/inventory/`, which Git ignores. Keep these
outputs private: `motions.jsonl.gz` retains all 51 original metadata fields,
including language annotations and actor metadata. `manifest.json` records the
source revision and metadata hash, motion counts, source archive hashes/sizes,
and the temporal-label object identity. It reports whether local file presence
was checked. It does not report files as downloaded, retargeted, or training-admitted.
Missing local paths remain explicit in each motion record, with aggregate counts
per representation. Existence is not byte-integrity or motion-quality evidence.

## Observed source, 2026-11-07

Authorized metadata download and complete inventory produced:

| Measurement | Value |
|---|---:|
| Metadata rows / columns | 142,220 / 51 |
| Original / mirrored motions | 71,132 / 71,088 |
| Original/mirror source groups | 71,132 |
| Unresolved mirror parent filenames | 0 |
| Actors / packages / categories | 522 / 8 / 20 |
| Metadata frames | 124,554,605 |
| Frame-hours at the documented 120 fps | 288.320845 |

Metadata SHA256:
`7cbe5624f91ae19af3feee69cff74b2cf553fb26367c8edc4b674c3d8ea97e7d`.
The tool verifies these bytes before processing and requires the complete pinned
row/column count. It has no sample limit or category filter.

Use **`filename`**, not `move_name`, as the identity and temporal-annotation join key:
those columns differ in 52,508 rows. Every mirrored filename has an `_M` suffix
and an original counterpart with the same actor, take and capture date. In 7,653
pairs the frame counts differ; equality of duration is not required for grouping.
The manifest keeps each original/mirror family together as `source_group` for a
later governed split. It does not assign a training or evaluation split. A split
that tests unseen actors should additionally group by `actor_uid`.

The actual v004 path field is **`move_g1_path`**. The dataset card's
`move_g1_mujoco_path` and v002/v003 examples are stale. All source representations
are retained in the inventory: SOMA Uniform BVH, SOMA Proportional BVH, G1 CSV,
and corresponding shape paths. The public API lists 529 proportional shape files;
522 actors occur in the motion metadata. Do not infer motion count from shape count.

## Retargeting and scale

For the Sentient humanoid, begin with the SOMA source and the corrected canonical
Sentient embodiment. G1 trajectories already embody another robot's geometry and
joint ordering. A successful source inventory does not establish retargeting,
full-path feasibility, contact validity, or simulation dynamics.

The training source is the human SOMA motion data, shape parameters and metadata.
Retarget directly from that source to the chosen canonical embodiment. The
repository mirror also retains `g1.tar.gz` because it copies the entire published
tree; that archive is an existing Unitree retarget and is not an input or an
intermediate step for Sentient retargeting. Excluding it, the human-source files
total 90,870,308,054 bytes. Storage completeness does not make every mirrored
representation a training input.

The three compressed motion archives total 114,156,145,800 bytes (114.16 GB).
`soma_uniform.tar.gz` alone is 45,188,088,297 bytes. Uncompressed storage and derived
references cost additional space. The current inventory does not fetch them.
Use bounded-memory conversion and training shards; loading this corpus as one
GPU array is a separate scaling problem from indexing its metadata.

Public viewer code at `bones-studio/seed-viewer@d9963c21bee413e7eb63c1b88ae01cdcb2cf3601`
describes a headered 120 Hz G1 CSV with root translation/rotation fields and joint
DOFs, interpreted as centimeters/degrees. That differs from xmimic's original
headerless LAFAN qpos input. The current archives have not been inspected, so no
converter is claimed from that viewer evidence alone. Read each SOMA BVH's frame
time and channel declarations when converting; do not silently reinterpret 120 Hz
source data as the LAFAN 30 Hz source.

Temporal labels are separately pinned as
`metadata/seed_metadata_v002_temporal_labels.jsonl`, SHA256
`302da3c319bbed4f712fa22a4d71b5288493911140cd72579e29de2ac324d634`.
These bytes are included in the private source mirror; joining them to training
examples remains separate work. The full-motion language descriptions in the
parquet are preserved. Future annotation admission must report missing and
unknown join keys and retain event timestamps against the source timebase.

## Access and verification

Access uses the dataset's [custom license](https://bones.studio/info/seed-license).
Existing account access does not make raw motions or metadata publishable in this
public repository. Neither the inventory tool nor a successful download determines
commercial eligibility or grants new rights. LAFAN retains its separate license
and provenance; combining sources does not erase either source's restrictions.

```bash
python -m unittest discover -s tests -p 'test_prepare_bones_seed.py' -v
```

These CPU checks cover deterministic completeness, original/mirror grouping with
unequal frame counts, unresolved parents, duplicate identities, unsafe paths and
explicit missing-file accounting. They do not train a policy or validate a robot.

## Private R2 custody

`tools/mirror_bones_seed.py` copies the complete pinned repository through the
SentientX Catalog SDK. Its source identity is 541 files and 114,370,281,701 bytes;
the normalized inventory digest also binds every path, size and source hash.
This includes all three archives, all metadata, shapes, skeletons, README and
license. It does not extract archives or admit motions to training.

Use an installed `sentientx` SDK and `huggingface-hub`, an authorized Catalog
project, an existing HF credential and the private authenticated repository-tree
plan. Keep the plan and transport directory outside version control. The run
uses the existing production SDK profile, or an `SX_API_KEY` supplied through
the secret environment; credentials never belong in command arguments.

```bash
python tools/mirror_bones_seed.py mirror \
  --project "$PROJECT_ID" --pack "$PACK_NAME" \
  --plan /private/mirror-plan.json --output /private/bones-transfer

# The receipt supplies the immutable manifest digest. No HF access is needed.
python tools/mirror_bones_seed.py restore \
  --project "$PROJECT_ID" --pack "$PACK_NAME" \
  --manifest-sha256 "$MANIFEST_SHA256" --output /private/bones-seed

python -m unittest discover -s tests -p 'test_mirror_bones_seed.py' -q
```

Catalog's worker scratch is smaller than the 45 GB source archives, so source
files are stored as ordered chunks of at most 1 GiB. Catalog independently
verifies each chunk before local transport scratch is pruned. The final private
pack includes every chunk and a reconstruction manifest, and is published only
after every complete source hash passes. LFS objects use SHA-256; ordinary Git
files use Git's blob-header SHA-1. Every reconstructed file also has SHA-256.
The manifest is read back before publication.

Mirroring runs four independent files concurrently, largest first, with separate clients and at
most one 1 GiB chunk of scratch per worker. Each file's hash remains sequential;
manifest ordering follows the pinned plan. Failures stop new work after active
chunks settle. Both commands retain checkpoints and reject changed identities, unsafe paths,
symlinks, corrupt retained bytes and incomplete inventories. Multipart ETags
survive interruption. Restore rechecks completed local chunks and resumes at a
verified boundary. Transient result-read failures permit two bounded
reattachments to the same durable operation using fresh clients. A failed Catalog
operation stops for inspection; it does not start another operation. A multipart
PUT retries only explicit HTTP 429/500/502/503/504 responses, at most twice, using
a fresh grant for the same upload ID, part number and byte range. Completed parts
are preserved. Other upload failures and registration writes stop for inspection.
Preserve the journal and receipt.

The pack is private internal source custody under the BONES-SEED license, not a
public dataset or a new ownership claim. Derived training selections retain the
source license and must separately enforce held-out membership and quality.
