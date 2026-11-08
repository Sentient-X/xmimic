# Managed runtime images

`Dockerfile.cpu` and `Dockerfile.gpu` package `xmimic.managed_worker:retarget` and
`xmimic.managed_worker:train` for the existing SentientX authored Job launcher.
Both expose `/opt/sx/.venv/bin/python` and install an exact local SentientX SDK
wheel containing `sx_training`, `sx_embodiments`, and `sx_geometry`. Robot assets,
source motion, API keys, registry credentials, and user configuration are excluded
from the build context. Governed input packs supply embodiment and dataset bytes.

The Python 3.12 Linux amd64 base is pinned by its platform manifest digest, resolved
from Docker Hub on 2026-11-08. The GPU recipe obtains CUDA 12 userspace from the
repository's existing JAX/NVIDIA wheel pins; Modal supplies the NVIDIA driver.
Its authored Job must declare `runtime_abi="cuda12"`. The CPU recipe declares
`runtime_abi="python3.12"`, accelerator `NONE`, and zero accelerator count/VRAM.

Prepare the actual unpublished runtime from the owning checkout (generated protocol
must already be current):

```bash
python tools/build_runtime_wheels.py --owner /absolute/path/to/glued-worktree
python tools/build_runtime_wheels.py --verify
```

This calls the existing owning `packages/sx/scripts/build_sdk_dist.py`; it does not
copy import roots or invent published internal package versions. The resulting
`sentientx-0.0.0+src.<sourcehash>` is a local wheel only. The SDK builder's asset audit
removes proprietary robot asset bytes. The ignored `data/runtime-images/manifest.json`
records every selected owner/XMimic source-file hash, superproject and embodiment
HEADs, package closure, wheel hash, and CPU/GPU dependency-lock hashes. The embodiment
revision comes from the superproject's mode-160000 gitlink, including archive-filled
worktrees that have no nested `.git`. HEAD alone
never identifies dirty source. Uncommitted and generated Python files are included
in the source identity. A changed source snapshot or wheel fails image verification.

The wrapper resolves Python 3.12 Linux amd64 binary dependencies through `uv pip
compile --generate-hashes`; images install those locks with `--require-hashes`.
Regeneration reuses an intact SDK wheel when its owner source identity is unchanged.
It never launches compute, builds an image, publishes a registry artifact, or retries
a paid Job. The resulting image must be published through its owning build/release
path, then admitted by immutable digest—not by tag. Publicly pullable authored images
are required by the current launcher; do not put private data or secret material in them.

## Observed build boundary

On 2026-11-08, installed `sentientx 0.2.6001+ge49f81e577a1` exposes
`sx source builds create` with Project, repository binding, succeeded checkpoint,
exact HEAD, attempt ID, fixed recipe, entrypoint and target platform. Its protobuf
accepts only `source.python-world/v1` and `source.python-component/v1`. Those recipes
build dependency-free pure Python components atop a trusted CPU runtime base; they
accept no Dockerfile, custom base, dependency-wheel bundle, or CUDA runtime.

A real read of Source's public `ListRepositories` succeeded for the Platform Project
and returned one repository; XMimic was not among the bindings. No build was submitted.
The existing Source recipes cannot build these runtime images without an owning
recipe extension. A trusted release image pipeline is another possible owner route,
but has not been wired or exercised for these images. Do not use an independent
Modal image builder to bypass this boundary.

Local evidence: the owning SDK builder produced an 18-package, approximately 4 MiB
wheel and excluded 2,505 robot asset files. Both CPU and CUDA 12 dependency closures
resolved for Linux amd64/Python 3.12. This is packaging evidence only: no container
build, published image, deployed authored-job traversal, GPU training, or physical
robot qualification is established by these checks.
