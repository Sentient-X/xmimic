# Native controls

`tools/benchmark_native.py --root data/native-baselines` acquires or verifies the exact
unmodified source archives, checkpoint bytes and task files. The acquisition receipt
contains every source-file hash; it contains no measured benchmark score.

Pinned sources are XMimic `8f7693f96ca02c49f11dcc98cd14cf03f45df2fa` and PolicyIK
`22f3a5cddc674e88995cae18b93cd5a5cb3ddb86`. PolicyIK releases three policies and
4,120 held-out OpenArm tasks. XMimic has no released checkpoint in this snapshot;
its control first trains the original 3,500-iteration G1/LAFAN task. Required G1
meshes and all 40 motion files must match upstream `tools/data_sha256.txt`.
These inputs are exclusively the requested native G1 control, never an intermediate
for Sentient retargeting.

The authored entrypoint is `tools.benchmark_native:run`. Its closed config is:

```json
{"method":"vanilla_policyik","phase":"evaluate_single","seed":1}
```

Methods are `vanilla_xmimic` and `vanilla_policyik`. Phases are `train`,
`evaluate_single`, and `evaluate_deployment`; PolicyIK uses its released checkpoints,
so native training is deliberately unavailable. XMimic has no PolicyIK deployment
protocol. Seeds 1,2,3 identify separate runs. PolicyIK deployment retains the original
single attempt, corrected goal and eight transient attempts separately.

Inputs are governed `source.tar.gz` and, for XMimic, `task/<upstream-relative-path>`
for every checksum-manifest file. XMimic evaluation additionally receives
`checkpoints/ckpt_*.npz` from its completed native training Job. The worker refuses
optimizer dispatch unless the manifest has training purpose and eligible ancestry;
evaluation requires evaluation purpose and false training eligibility.

The worker runs unchanged upstream modules in a persistent working copy and writes
stdout, native checkpoints and results under the managed run directory. It records
intent before each child starts, verifies completed outputs before continuing, and
never automatically repeats an interrupted or ambiguous paid command. Native upstream
training does not provide complete optimizer/simulator recovery here: retain failed
attempt evidence rather than claiming portable checkpoint recovery or buying it twice.

Runtime requirements remain explicit. XMimic needs its pinned CUDA12/JAX stack;
PolicyIK needs Python3.11 at `/opt/policyik/.venv/bin/python`, IsaacSim5.1,
IsaacLabv2.3.2, Torch2.7/CUDA12.8 and pinned OpenArm sources at
`/opt/policyik/openarm_isaac_lab`. Its original scripts accept the NVIDIA Omniverse
license. NVIDIA's [IsaacSim5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
exclude GPUs without RT cores, including H100/A100; select a supported RTX-capable
managed allocation such as L40S. A different simulator is an adaptation, not vanilla.

Twenty-nine local tests cover source corruption, archive traversal/link refusal, distinct
native protocols, training/evaluation intent, actual child-process restart refusal, and
real `TrainingContext.execute` artifact custody. Completed output is copied atomically
to the required `run_dir/artifact`; publication interruption resumes from verified
completed steps without restarting native computation. Temporary writes bind the
destination name and content digest, and exact-byte reconciliation covers interruption
after fsync both before and after publication linking. Unknown temporary files refuse. Source and output inventories
reject unrecorded files and links. Stop handling kills the whole process group even
when its leader exits before a SIGTERM-ignoring descendant.
They are not native simulation results. Runtime publication, managed dispatch and
completed native scores still require their own receipts.


`tools/prepare_native_inputs.py` prepares the exact archived-source/task closure and
publishes it through the existing private Catalog asset-pack client. It is not a
launcher. Pass `--root data/native-baselines`, `--method vanilla_xmimic` (or
`vanilla_policyik`), a persistent `--output`, the reviewed runtime's
`--producer-revision`, and `--publish-project`. The producer revision identifies the
worker/input contract; upstream revisions and archive checksums remain in
`native-acquisition.json`. The preparation tool's source revision is recorded by git.
The publication journal retains each accepted operation, so an ambiguous failure
must be reconciled before retrying; never discard that journal.

On 2026-11-08, the private project published the 77-file XMimic training closure as
`native.vanilla_xmimic.336e2cb7281ece7c180313aa` and the two-file PolicyIK evaluation
closure as `native.vanilla_policyik.48eb8797fd56ac0bff202bdb`. Both manifests were read
back through the public authenticated Catalog client and matched byte-for-byte.
`data/native-baselines/native-input-publication.json` retains their full digests.
This proves input custody, not completed native training or evaluation. Submit only
through the existing authored training Job API/CLI with the correct purpose,
immutable image, explicit idempotency key, deadline and spending cap.

## First managed launch evidence

The production dispatcher was qualified on source `1061ded` with the exact private
runtime grants and migration `0019`. H100 Job
`ftjob_4c0c0bece5664eedb6eae8f3c922c496` was accepted on 2026-11-08 at 16:33:57 UTC,
with an $8 cap, 3,600-second deadline and the original 3,500-iteration protocol.
Its retained attempt ended failed at 16:50:03 UTC. No upstream training command or
optimizer ran, and no native score was produced.

The runtime's stderr rejects symlinks above the staged artifact root before source
access. Temporal history independently records a 120-second launch heartbeat
timeout; the old dispatcher then repeated the bootstrap failure through three
recovery generations. The public Watch reader also refused an inconsistent failure
snapshot. The stored failure itself is complete (`execution_failed`, nonretryable).
The exact provider symlink component could not be inspected after termination.

Retained evidence is under
`data/native-baselines/runtime-grants-rollout-v1/`: sanitized Temporal history,
native attempt facts, diagnosis, and private provider logs. The reviewed corrections
address trusted mount boundaries, slow-launch custody, explicit checkpointed recovery
and atomic operation snapshots. Corrected hosted execution remains unverified until
new validated immutable runtime images and their exact grants are serving. Preserve
this failed Job and its costs; do not silently relabel or resubmit its idempotency key.

After a native XMimic training Job succeeds, run the same preparation command with
`--training-job-id <retained Job ID>`. The helper reads that Job and its immutable
owner result, verifies the original input pack, complete 35-checkpoint inventory and
native/owner receipts, and composes an evaluation-only pack from existing Catalog
references. It downloads only the two manifests and two small proof documents;
checkpoint bytes stay in Catalog custody. Only the new evaluation manifest is uploaded.
The private pack retains lineage to both original inputs and training outputs.

This handoff requires Catalog's asset-pack `ReadAsset` capability and the SDK's direct
`AssetRef` registration support. Its local `composition.json` journal records intent
before pack registration and lineage writes. Lost acknowledgements reconcile against
exact public readbacks; absent or conflicting state refuses rather than resubmitting.
Keep the original input publication journal and the composition directory. Seventeen
local helper tests cover integrity, full-checkpoint refusal and acknowledgement-loss
recovery; hosted composition and native evaluation still need their own receipts.

Managed volume roots may be provider aliases. Native custody uses the shared
`sx_training.paths.rooted_path` check below `context.run_dir`, preserving the logical
path in command journals. A different physical target for the same remounted volume
must not change command identity or rerun a completed child. Links inside the staged
source, output or checkpoint closure remain refused. The runtime image must contain
this same SDK contract; its installation proof reads an admitted file through linked
mount roots before any native GPU attempt is made.

## Corrected runtime and publication evidence

On 2026-11-08, corrected CPU and CUDA runtime publication 42506620271 produced
`sx-xmimic-cpu@sha256:9923109ca044b6952df1abf3d92e381ae4522a8fec9cf6ead89b38e0bfadddef`
and
`sx-xmimic-cuda12@sha256:0a448e7a9cfc64be0ef1dd89f39edb087cae0067b4eeed2372ba26b35bd44d1f`
under `registry.sentientx.io/sx/cloud/`. Both contain native worker `20f067b` and
passed the installed exact-mount behavior proof.

The 19:18 UTC live acceptance verified all 11 reviewed module hashes across the
ready API/dispatcher roles, both exact project grants, schema 0019/JSONB and an
authenticated Catalog ReadAsset of the original native input manifest. The serving
API image was `6200697`, source `217b321b`. Receipt:
`data/native-baselines/runtime-correction-v2/serving-20261108T191832Z.json`.
Fresh containment found no unfinished Sandbox for the failed original Job. The
separate corrected Job `ftjob_73935460dead4ce68eb83b4aaf327a36` was accepted at
19:19:20 UTC with the same original 3,500-iteration seed1 protocol, one H100,
$8/3,600-second limits. Acceptance and RUNNING do not establish device execution,
native training or a score. Cold placement has retained one activity attempt and
recent heartbeats across the earlier 120-second failure boundary, with no recovery
generation. Its immutable intent and watch evidence remain under `submissions/`.

Native PolicyIK required the actual PEP517 editable metadata of all six pinned
source packages. The legacy rl-games setup.py list omitted its authoritative
Poetry backend and gym0.23/psutil5/ray/wandb requirements. The owning runtime now
regenerates and compares source/backend metadata on Linux before installation,
keeps the vendor SDK unchanged and checks the complete isolated dependency graph.
Its 161 locked dependencies plus 6 source distributions passed native imports and
the managed mount/artifact proof in candidate 42511520865. That candidate pushed
its image but failed required registry metadata validation; it is retained as
failed-publication evidence, not an admitted runtime.

The shared image generator correction supplies all three missing inventory-owned
OCI labels for built targets while preserving exact upstream mirrors. Changed
publication 42515309040 passed actual Zot metadata readback and produced:

`registry.sentientx.io/sx/cloud/sx-policyik-runtime@sha256:d59c2c61396d22387a1a5aeb5789eb92509aabbaf654c60dda165db4ebe6201c`

Source is `e353302380`. The unchanged native/import/SDK proof layers were reused
from durable cache: layer push 11.8s, manifest 21.4s, publication step 44.641s. The
first push had taken 1,315.7s after 204.8s export. These are build/publication times,
not model execution times. Full logs and final receipt are retained under
`data/native-baselines/runtime-metadata/policyik-pep517-v1/publication-result.json`.
This is a private research runtime publication, not production security or GPU
simulation qualification. Exact project authorization, live configuration and the
original L40S evaluation still require their own evidence. Do not use the earlier
failed-publication digest or infer native scores from these installation checks.
