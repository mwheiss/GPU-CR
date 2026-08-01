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

Set `EXPORT_FILE_PATH` to a directory on a regular filesystem. The primary
checkpoint and bounce-buffer files are mapped `MAP_SHARED`. The primary file
is pageable and may be written back and reclaimed by Linux. Only the two
transfer bounce buffers are registered with CUDA as pinned memory.

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

File capacity is reserved with `posix_fallocate()` before mmap by default so a
checkpoint cannot fail late with `SIGBUS` or `ENOSPC`. Set
`GPU_CR_FILE_PREALLOCATE=0` only when sparse allocation is explicitly desired.
This does not call `fsync`; normal checkpoint completion means the image is in
the shared filesystem mapping and may still contain dirty page-cache pages.

## NVIDIA control checkpoint

Use NVIDIA's current upstream `cuda-checkpoint` binary matching the installed
driver. For driver 610 checkpoint jobs, lock every member before checkpointing
any member, then checkpoint, restore and unlock processes sequentially in a
stable order. `multi_cr_client` follows that ordering on this branch.

For the BetterIssa single-GPU vLLM profile, the expected sequence is:

```text
quiesce requests
GPU data -> mapped file and release physical allocations
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
