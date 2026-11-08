# Persistent retarget artifacts

`xmimic.artifact_cache` retains CPU retarget results locally and publishes completed
inventories through the released Catalog SDK. It launches no computation. A cache
hit establishes exact byte reuse, not motion quality or physical qualification.

A `CacheIdentity` binds seven SHA-256 values: the exact source inventory, shape
inputs, embodiment document, URDF bytes, mapper implementation/configuration,
solver implementation/dependencies, and settings. Use `content_hash(settings)`
for canonical finite JSON settings. Include all physically relevant projection,
joint ordering, timebase and solver options in these identities. A filename,
mutable branch or robot family name is insufficient. The caller must compute
these facts from actual inputs; the cache cannot inspect a running solver.

```python
from pathlib import Path
from xmimic.artifact_cache import ArtifactCache, CacheIdentity

# Identity values come from the source/target adapter's verified manifests.
identity = CacheIdentity(**identity_fields)
with ArtifactCache(Path("~/.cache/xmimic/retarget")).entry(identity, clip_ids) as entry:
    # Inspect entry.document first. Never rerun a retained running clip.
    entry.start_clip(clip_id, execution_ref)
    entry.settle_clip(
        clip_id, execution_ref, status="succeeded",
        artifacts={"motion.npz": Path("output/motion.npz")},
        metadata={"quality_status": "unqualified"},
    )
    # Rejected/failed clips use their own terminal status and a nonempty reason.
    # Complete only after every clip in clip_ids has a terminal disposition.
    manifest_sha256 = entry.complete()
```

The entry context holds a process lock. Writes use fsync and atomic replacement;
output blobs are immutable by content hash. Reopening verifies retained manifests
and output hashes. Corruption is refused, never silently repaired or recomputed.
Every source clip stays in the inventory, including pending, running, rejected and
failed clips. `complete` means all dispositions were retained; it does **not** mean
all clips succeeded or that the dataset is eligible for training. Training selection
must inspect the dispositions and its own quality/held-out policy.

An interruption after `start_clip` leaves the original execution reference running.
Only that reference may settle the clip. `retarget_clip` creates a bound frame journal
before the solver starts, so the same execution can resume its committed prefix under
the cache lock. The last frame's exact coordinates warm-start the next frame; rejected
frames retain their diagnostics and count as committed progress too. Each complete
frame and its separate count/hash tip are fsynced before the next frame begins. A
missing committed tail is refused even when the platform checkpoint is older. A
complete frame ahead of an interrupted tip write is verified and adopted. A kill during an unfinished frame may
repeat that uncommitted frame, but acknowledged frames are never solved again.

Legacy running entries without a frame journal still require reconciliation; there is
no failed-clip reset. Missing or corrupt state is refused rather than recreated. The
managed CPU worker checkpoints a journal receipt every 32 frames, at completion and
on cooperative stop. On replacement it verifies the acknowledged prefix against the
same retained run-volume journal. Native progress may be ahead of that receipt, never
behind it. The receipt is not a replacement for missing coordinates. Local process-kill
and managed-runtime tests establish this behavior; Modal volume visibility across
actual hosted eviction requires separate execution evidence.
An interrupted file copy can leave an unreferenced blob; it never becomes a success.
A crash between committing the completed manifest and state can finish the same
manifest on reopening. Do not remove cache state to force a paid job to start again.

## Catalog custody and restore

Construct a signed-in `CatalogClient` through the installed SentientX SDK, then use
`CatalogPublisher(client, project_uuid, Provenance(repository, revision, license_id,
generator))`. Supply the actual retarget result's licensing and provenance; neither
BONES-SEED source provenance nor the code license automatically describes derived
motion outputs. `publish(entry, pack_name)` requires the entry lock and a complete
inventory. It publishes every blob and the manifest in an immutable private asset
pack. Retain its returned Project, pack and manifest SHA as the restore receipt.

Files must be nonempty and at most 1 GiB. Shard retarget outputs before publication
when larger; this library does not introduce another archive/chunk format. An empty
source file can be represented as metadata without a zero-length Catalog upload.

Publication persists multipart IDs, part ETags and each accepted Catalog operation
reference. A later invocation reattaches to those exact operations; it never starts
paid computation. Network or provider errors stop the call with local evidence intact.
No mutation is retried in the same call. The immutable pack's membership can be
replayed explicitly after a lost pack acknowledgement.

A crash after registration submission but before receipt persistence leaves
`registering` without `operation`. Publication refuses to resubmit. Recover the
operation through Catalog, then call
`reconcile_registration(entry, object_sha, operation_ref, resource_ref)`. This reads
and verifies the exact operation's resulting SHA before updating the local journal.
Do not invent an operation reference or edit the journal by hand. Multipart PUT
response loss permits resending the same unacknowledged part of the same upload;
it does not create a new logical computation or artifact.

`restore(cache, pack_name, manifest_sha256)` requires only Catalog access. It checks
the manifest identity and every object hash, preserves downloaded evidence on
failure, and refuses to overwrite an existing partial run or corrupt blob. It does
not contact the original motion provider. No bucket credential or direct R2 path
is accepted. Signed grants are used in memory, never stored in the journal.

## Remaining owning execution changes

Remote CPU retargeting and Modal GPU training belong to the managed execution
owner, not a standalone Modal launcher in XMimic. The inspected authored path needs:

- A typed immutable artifact-pack input alternative to `TrainingCorpus`, resolved
  and frozen at admission, then staged through Catalog with manifest-last commit.
- An honest CPU authored execution shape (`gpu=None`, CPU runtime ABI and CPU cost
  accounting); the current authored admission requires one GPU and Modal projection
  assumes CUDA. Preserve public image pins, isolated execution and no SX credentials.
- Generic artifact publication with source lineage for retarget outputs, without
  calling preparation a corpus-trained checkpoint. The GPU learner consumes that
  exact artifact input. Training use needs the Catalog-owned schema/parser and
  held-out ancestry admission; a storage pack alone is not training permission.
- Regenerated/released SDK and owner images, followed by bounded CPU and Modal GPU
  execution/recovery evidence. The inspected installed SDK
  `0.2.6001+ge49f81e577a1` has no `AuthoredTrainer` field.

The files that own those changes are Train's `jobs/model.py`, `connect.py`,
`catalog.py`, `document.py`, `data/bundle.py`, `launch/modal.py`, and `jobs/cost.py`,
the training proto, Catalog training-use admission, and `sx_training`. Existing
`WorkspaceJob` profiles are reviewed domain executables, not an arbitrary job API.

## Verification and failure lessons

```bash
python -m pytest tests/test_artifact_cache.py -q
```

These are local integrity, process-lock and restart tests, including a local byte
reader for restore. They do not simulate or qualify a deployed Catalog service.

| Failure | Retained behavior |
|---|---|
| Original work was running at interruption | Same execution and verified frame journal resume the committed prefix; legacy ambiguity refuses |
| State, manifest or artifact bytes changed | Refuse; preserve evidence |
| Blob committed before state write failed | State remains running; original execution can settle |
| Manifest committed before completion flag | Finish the exact same manifest |
| Registration acknowledgement lost | Reconcile exact Catalog operation; no resubmission |
| Remote member corrupt during restore | Preserve partial cache and refuse completion |
