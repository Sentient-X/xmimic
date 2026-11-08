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
Keep the original input publication journal and the composition directory. Sixteen
local helper tests cover integrity, full-checkpoint refusal and acknowledgement-loss
recovery; hosted composition and native evaluation still need their own receipts.
