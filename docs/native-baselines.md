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

Nine local tests cover source corruption, archive traversal/link refusal, distinct
native protocols, training/evaluation intent and actual child-process restart refusal.
They are not native simulation results. Runtime publication, managed dispatch and
completed native scores still require their own receipts.
