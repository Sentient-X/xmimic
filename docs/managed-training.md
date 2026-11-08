# Embodiment → retarget cache → managed training

The fork supplies an authored CPU retargeter and an authored single-H100 learner.
SentientX owns placement on Modal, credentials, deadlines, spend caps, persistent
volumes and Catalog/R2 publication. Robot geometry comes from `sx-embodiments`.
Human SOMA data goes directly to that robot; G1 motion is never an intermediate.

`tools/prepare_embodiment_job.py` prepares a portable canonical body, its verified
asset closure, an explicit nominal simulation profile, a SOMA clip and a retarget
recipe. A new morphology requires its own landmark/segment mapping and profile;
the code never infers joint semantics from tensor width or silently substitutes G1.
The current projection supports a floating root and independent scalar revolute
position-controlled joints. Unsupported topology is refused explicitly.

`tools/managed_train.py` publishes this input directory privately, starts the CPU
Job through `Sx.train_custom`, downloads its immutable output into a local cache,
and submits successful motion inputs to the GPU Job. Example invocation after
preparing inputs and publishing compatible runtime images:

```bash
python tools/managed_train.py \
  --project PROJECT_UUID --inputs /absolute/path/prepared \
  --source-entry CATALOG_SOURCE_PACK --embodiment-id EXACT_BODY_SHA256 \
  --clip-id ORIGINAL_CLIP_ID --source-group ORIGINAL_MIRROR_FAMILY_ID \
  --cpu-image REGISTRY/CPU_IMAGE@sha256:DIGEST \
  --gpu-image REGISTRY/GPU_IMAGE@sha256:DIGEST \
  --revision XMIMIC_COMMIT --license-id LicenseRef-BONES-SEED \
  --training-config /absolute/path/training.json \
  --run-dir /absolute/path/persistent-run \
  --cache-dir /absolute/path/persistent-cache
```

To select a different embodiment in the same workflow, replace `--inputs` and
`--embodiment-id` with:

```bash
--embodiment REGISTERED_NAME \
--source /absolute/path/complete-soma-clip.bvh \
--nominal-profile /absolute/path/nominal.json \
--recipe /absolute/path/retarget-recipe.json \
--training-profile /absolute/path/training-profile.json
```

For a pinned development composition use `--embodiment-document body.json` instead
of `--embodiment`. Preparation resolves the body through `sx-embodiments`, retains
its complete asset closure under the run directory, and derives its identity; no
second manually entered body ID is required. Reopening the same run verifies the
same body/source/profile binding. A changed selection requires a new run directory.
Use `--profile preview` for the preview platform. Both stages return ordinary
training operation references usable with `sx operation watch`, `status`, `result`
and `cancel`.

The platform's `sx training custom start --authored-trainer trainer.json
--artifact-inputs '<ArtifactPackInputs JSON>' --execution '<TrainingExecution JSON>'`
exposes the same contract directly. The fork owns this workflow's human-to-robot
mapping; the platform owns admission and Modal execution for every embodiment.

The default bounds are ten minutes/$1 for CPU and ten minutes/$5 for GPU. They are
limits, not runtime or cost predictions. Reinvoke with the same run directory and
settings to reattach to the same operations. A changed request refuses reuse of a
paid stage. The workflow does not cancel work when the local terminal disconnects.

The CPU stage retains every frame and reports rejected clips. Only whole accepted
clips produce a training candidate; rejection diagnostics never become a motion
input. Catalog admits the selected input closure and checks held-out source ancestry
before scheduling learning. A family identifier alone is not Catalog partition
admission. Use a source entry at the intended membership granularity: parenting all
clips to one archive conservatively couples that archive's evaluation ancestry.

Caches bind source/shape, exact embodiment/URDF, mapping, solver code/version and
settings. Upload journals retain multipart state and accepted operation references;
an ambiguous registration acknowledgement requires reconciliation, not a fresh
paid run. Intermediate recovery checkpoints live in the managed run volume. Final
output includes the policy and complete recovery state in checksummed ≤1 GiB chunks
on Catalog/R2, which the workflow also restores locally. Recovery binds code,
configuration, source, robot and runtime; policy-only NPZ files cannot resume training.

The user-approved Sentient profile is **nominal simulation only**. It changes no
physical drive facts and qualifies neither balance, collision safety nor hardware
operation. Geometric retargeting tests and exact CPU/JAX continuation are separate
from actual Warp/GPU recovery and policy evaluation.

## Measured limits

The published upstream benchmark reports a mean2.48 minutes on one H100 NVL and
0.91 minutes on four H100 NVLs for40 G1/LAFAN motions, excluding compilation. It is
not a one-minute guarantee for another embodiment or the entire BONES-SEED corpus.
Source: [upstream results](https://github.com/nico-bohlinger/xmimic#results).

The current learner loads the selected reference frames into GPU memory. Select a
bounded motion pack; whole-corpus residency and multi-GPU authored placement are not
implemented by this workflow. The raw BONES-SEED repository is already mirrored
privately to R2; downloading it again from Hugging Face is unnecessary.

The owning platform changes and [runtime image recipes](runtime-images.md) require
publication before this command can execute remotely. Local tests do not establish
a deployed Modal run, a trained Sentient policy or physical acceptance.


Local acceptance on 2026-11-08:178 CPU tests and44subtests passed. Independent review
found and corrected output-directory and interrupted-materialization bugs; its20-check
correction pass retained no finding. A real rejected599-frame retarget cache was
published privately to Catalog/R2, linked to its mirrored source, and downloaded with
all member digests verified. It remains excluded from training. Two complete source
clips have been tested; neither passed the current Sentient geometric profile.
