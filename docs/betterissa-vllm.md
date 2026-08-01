# BetterIssa vLLM checkpoint profile

This branch carries a small, upstreamable compatibility layer for persistent
vLLM API servers. It does not serialize or terminate the CPU process.

## Control safety

Checkpoint signals only write their signal number to a pipe. A dedicated
thread performs initialization, CUDA calls, allocation, mmap access and
coordinator communication. The controller is recreated from an allocation
hook after `fork()`, because only the calling thread survives a fork. This is
required by vLLM V1 when its EngineCore worker uses the `fork` start method.

## File-backed memory

Set `EXPORT_FILE_PATH` to a directory on a regular filesystem. Only the two
transfer bounce buffers are registered with CUDA as pinned memory.

With `GPU_CR_ASYNC_PERSIST=1`, checkpointing uses two host tiers:

- a prefaulted anonymous `mmap` receives the synchronous GPU copy;
- `ckpt-N.bulk.data` receives an aligned background `O_DIRECT` mirror.

VRAM is released as soon as the first copy finishes. A restore locks the
current persistence boundary and reads each chunk from RAM or disk, so it can
preempt persistence without observing a mixed generation. Starting another
checkpoint cancels the obsolete mirror at a chunk boundary. The coordinator's
`ckpt-N.data` remains mapped, but only its first 2 MiB is physically allocated;
the unused bulk range is replaced with an inaccessible address reservation.

`GPU_CR_CHECKPOINT_CACHE_POLICY=keep` is appropriate for a frequently restored
primary model. It retains the RAM copy and uses the bulk file as recovery
backing. `pageout` is appropriate for a rare fallback: every RAM chunk receives
`MADV_DONTNEED` after the direct write completes, so a later restore reads the
bulk file at the volume's sustained direct-read rate. The backing filesystem
must support aligned `O_DIRECT`; failure is reported in
`ckpt-N.persist.json` and should trigger the manager's cold-restart fallback.

The bounce-buffer size is selected at build time:

```bash
cmake -S . -B build \
  -DGPU_VENDOR=NVIDIA \
  -DCUDA_ROOT=/opt/cuda \
  -DSHM_SIZE_GB=25 \
  -DSTAGING_BUF_MB=256
cmake --build build -j
```

Build with a userspace ABI no newer than the target vLLM container. BetterIssa
uses `nvidia/cuda:13.0.2-devel-ubuntu22.04` (GCC 11, glibc 2.35) for both the
preload library and coordinators. Static libstdc++ linking does not solve a
newer host glibc dependency; `GPU_CR_STATIC_CXX_RUNTIME` therefore defaults
off and is only an explicit packaging option.

File capacity is reserved with `posix_fallocate()` before use. In two-tier mode
the control file reserves 2 MiB and the bulk sidecar reserves exactly the
meaningful allocation image; the legacy synchronous mode reserves the complete
mapped image. Set `GPU_CR_FILE_PREALLOCATE=0` only when sparse allocation is
explicitly desired. GPU-CR does not call `fsync`. A completed direct write is
available for restore, but power loss can still require a cold start and a new
checkpoint.

## NVIDIA control checkpoint

Use NVIDIA's current upstream `cuda-checkpoint` binary matching the installed
driver. For driver 610 checkpoint jobs, lock every member before checkpointing
any member, then checkpoint, restore and unlock processes sequentially in a
stable order. `multi_cr_client` follows that ordering on this branch.

For the BetterIssa single-GPU vLLM profile, the expected sequence is:

```text
quiesce requests
GPU data -> RAM mmap, release physical allocations, direct background mirror
CUDA lock -> checkpoint
CUDA restore -> unlock
remap allocations and mapped file -> GPU
real inference readiness check
```

The application manager must validate process identity, VRAM availability and
inference output around every transition. A failed partial transition must be
rolled back or cold-restarted; a checkpoint error is not safe to ignore.

`GPU_VENDOR` may be set explicitly in the managed process environment. When it
is absent, the runtime now selects the backend compiled into the preload
library (NVIDIA for a CUDA build, AMD for a HIP build). This matters for
services launched through an environment-sanitizing supervisor: selecting the
already-fixed build target must not terminate the long-lived worker during its
first checkpoint initialization.

The preload resolver must obtain wrapped CUDA VMM functions with `RTLD_NEXT`.
Resolving `cuMemCreate`, `cuMemUnmap`, or `cuMemRelease` through
`RTLD_DEFAULT` can select the preload library's own exported wrapper and cause
recursive controller calls during checkpoint teardown.

All GPU-CR control messages use dedicated realtime signals on this branch.
Persistent Python services, including vLLM, may install their own
`SIGUSR1`/`SIGUSR2` handlers after `LD_PRELOAD` constructors run; using those
traditional user signals can therefore terminate a worker or leave the
coordinator waiting forever after IPC teardown.

Coordinator waits are bounded by `GPU_CR_CONTROL_TIMEOUT_MS`, falling back to
`GPU_CR_LOCK_TIMEOUT_MS` and then 600000 ms. A dead worker fails immediately;
a lost control message cannot leave the model manager blocked indefinitely.
