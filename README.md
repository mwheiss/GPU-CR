<h1 align="center">GPU-CR: GPU Checkpoint & Restore</h1>

> The `codex/betterissa-mmap-control` branch adds a fork-safe control thread,
> bounded pinned bounce buffers, file-capacity reservation and driver-610
> sequential job ordering for persistent vLLM servers. See
> [docs/betterissa-vllm.md](docs/betterissa-vllm.md).

[![cuda](https://img.shields.io/badge/CUDA-supported-brightgreen.svg?logo=nvidia)]()
[![rocm](https://img.shields.io/badge/ROCm-supported-brightgreen.svg?logo=amd)]()
[![ascend](https://img.shields.io/badge/Ascend-Developing-lightgrey.svg?logo=huawei)]()

<div align=center><img width = '150' height ='150' src ="./source/GPU-CR.png"/></div>

GPU-CR is a system designed to support efficient Checkpoint and Restore (C/R)
for GPU-accelerated applications. Its key advantage is completely yielding the
GPU memory of the checkpointed app (reducing VRAM usage to 0), seamlessly
freeing up space for other workloads to swap in and execute.

![CLI Demonstration](./source/GPUCR_VS_CUDA.gif)
<p align="center">
  <em>Single-GPU CLI demonstration of the GPU-CR tool.</em>
</p>


## News

- **[2026-05]** Multi-GPU C/R lands on NVIDIA: phased `multi_cr_client`
  orchestrator, NCCL plugin adapter, zero-patch NVSHMEM examples, and a 4-model multi-GPU vLLM
  benchmark — see §II.3.
- **[2026-03]** Initial single-GPU release for **NVIDIA (CUDA)** and
  **AMD (ROCm)** — transparent `LD_PRELOAD` interception, `cr_client` CLI,
  hugepage-accelerated VRAM staging.


## I. Features

- **Cross-Vendor Support**: NVIDIA (CUDA) and AMD (ROCm), single-GPU.
- **Multi-GPU C/R (NVIDIA)**: Phased multi-process orchestrator
  (`multi_cr_client`), cuMem IPC/VMM teardown, optional
  [NCCL adapter](adapters/nccl/README.md) and
  [NVSHMEM examples](adapters/nvshmem/README.md).
- **Transparent C/R**: `LD_PRELOAD` injects a `vGPU` library that intercepts
  memory allocations and resource management — no application changes for the
  single-GPU and signal-driven multi-GPU paths.
- **Client CLI**: `cr_client` (single-GPU) and `multi_cr_client` (multi-GPU)
  trigger checkpoint and restore operations.
- **Performance Optimization**: Huge Pages support to accelerate VRAM staging.


## II. Performance Evaluation

We compare GPU-CR with existing GPU checkpoint solutions on four LLM workloads:

- Llama-8B
- Phi-4-mini-instruct
- pythia-1b
- Qwen3-1.7B

For GPU-CR, the latency is split into:

- Data — GPU data buffers
- Control — GPU control states

Total latency = Data + Control

### 1. NVIDIA Single-GPU (CUDA Checkpoint vs GPU-CR)
- **GPU:** NVIDIA A100-PCIE-40GB
- **Driver Version:** 580.95.05
- **CUDA Version:** 13.0
- **vLLM Version:** 0.14.1

![Performance Comparison](./source/gpu-cr_cuda.png "NVIDIA (CUDA Checkpoint vs GPU-CR)")

### 2. AMD Single-GPU (CRIU vs GPU-CR)
- **GPU:** AMD Instinct MI100
- **ROCm Version:** 6.4.3
- **vLLM Version:** 0.11.1-rc7

![Performance Comparison](./source/gpu-cr_criu_amd.png "AMD (CRIU vs GPU-CR)")

### 3. NVIDIA Multi-GPU (vLLM TP=1 PP=2, gpu_util=0.9)

- **GPUs:** 2 × NVIDIA A100-PCIE-40GB
- **Driver Version:** 580.95.05
- **CUDA Version:** 12.9
- **vLLM Version:** 0.14.1

![Multi-GPU Performance](./source/gpu-cr_cuda_multi_gpu.png "NVIDIA Multi-GPU (vLLM TP=1 PP=2, gpu_util=0.9)")

Times are dominated by the per-worker data plane (≈ 36 GiB GPU→host copy
during checkpoint, host→GPU during restore), so they barely depend on the
model — at `gpu_memory_utilization=0.9` vLLM fills the KV cache to roughly
the same size regardless of weight count.

> **Reproducing**: build with `-DSHM_SIZE_GB=40` (otherwise the default
> 25 GiB staging ceiling is hit when each worker dumps ≈ 36 GiB), reserve
> ≥ 45000 2 M hugepages (`echo 45056 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages`),
> mount `/mnt/huge-ckpt` with `size=88G`, then run
> `apps/vllm/launch_multi_gpu.sh` after `export VLLM_GPU_UTIL=0.9`.


## III. Prerequisites

- **Operating System**: Linux (tested on Ubuntu 22.04).
- **Build Tools**: CMake ≥ 3.18, GCC/G++, Make.
- **Checkpoint Backend & Drivers**:
  - **NVIDIA**:
    - CUDA Toolkit 12.x or later.
    - Uses `cuda-checkpoint` (**included in this repository**, under
      `cuda-checkpoint/`).
    - For the multi-GPU path: a recent driver supporting
      `cuda-checkpoint --action`.
    - For the NCCL adapter: an NCCL source tree (≥ 2.28 master) to patch
      (`adapters/nccl/nccl_patches/`).
    - For the NVSHMEM examples: an NVSHMEM install with header + libraries.
  - **AMD**:
    - ROCm 6.x or later.
    - A custom-built `criu` with the AMD plugin enabled (manual compilation
      required, not included in this repo).
      See the [CRIU AMDGPU Plugin docs](https://github.com/checkpoint-restore/criu/blob/criu-dev/plugins/amdgpu/README.md).


## IV. Building

Pick **one** vendor per build directory.

### Option 1: NVIDIA (CUDA)
```bash
mkdir build && cd build
cmake -DGPU_VENDOR=NVIDIA ..
make -j$(nproc)
# Outputs: vGPU-NVIDIA.so, cr_client, multi_cr_client
```

### Option 2: AMD (ROCm)
```bash
mkdir build && cd build
cmake -DGPU_VENDOR=AMD ..
make -j$(nproc)
# Outputs: vGPU-AMD.so, cr_client
```

### Option 3: NVIDIA + NCCL adapter
```bash
mkdir build && cd build
cmake -DGPU_VENDOR=NVIDIA \
      -DGPU_CR_BUILD_NCCL_ADAPTER=ON \
      -DNCCL_ROOT=/path/to/nccl ..
make -j$(nproc)
# Additional outputs:
#   adapters/nccl/libgcr_preload.so
#   adapters/nccl/libnccl-checkpoint-gcr.so
#   adapters/nccl/two_proc_nccl_test
```

`-DGPU_CR_BUILD_NVSHMEM_ADAPTER=ON -DNVSHMEM_ROOT=/path/to/nvshmem` adds the
NVSHMEM example binaries; the adapter ships no patch since the cuMem hooks
already cover NVSHMEM.

To run multi-GPU at `gpu_memory_utilization=0.9`, also pass
`-DSHM_SIZE_GB=40` so the per-worker staging buffer fits the larger dump.


## V. Usage

### 1. Environment (single-GPU and multi-GPU)

```bash
# Optional: file-backed persistent VRAM checkpoint image (instead of hugepages)
export EXPORT_FILE_PATH=/path/to/save/vram_dump_path

# Recommended for speed: reserve hugepages
sudo bash -c 'echo 40960 > /proc/sys/vm/nr_hugepages'   # ~80 GiB
sudo mkdir -p /mnt/huge-ckpt
sudo mount -t hugetlbfs nodev /mnt/huge-ckpt
sudo chmod 777 /mnt/huge-ckpt

# AMD-specific: where CRIU writes its checkpoint files
export AMD_CKPT_DIR=/path/to/save/criu_files
```

`EXPORT_FILE_PATH` affects only the persistent GPU-allocation image.  CUDA
copies pass through a separate anonymous `mmap` transfer buffer, which GPU-CR
registers as pinned host memory.  Keeping that temporary buffer on the regular
filesystem can make `cudaHostRegister` fail and reduce otherwise memory-local
restore throughput below the storage device's read rate.

Pinned staging is required by default.  Initialization exits if host-memory
registration fails so a production service cannot silently enter the known
slow pageable path.  Set `GPU_CR_ALLOW_PAGEABLE_STAGING=1` only as an explicit
compatibility fallback; it may severely degrade checkpoint and restore times.

The file-backed image uses `MAP_SHARED`.  Linux may write dirty pages to the
configured filesystem while retaining the resulting clean pages in page cache,
so hot restores can remain memory-speed and the pages can still be reclaimed
under pressure.  GPU-CR does not call `fsync`; applications that require
power-loss durability must add their own persistence barrier.

### 2. Single-GPU run + checkpoint

```bash
# Run the app under the preloader
LD_PRELOAD=$PWD/build/vGPU-NVIDIA.so python3 ./apps/vllm/serving_vllm_nvidia.py &
APP_PID=$!

# Checkpoint
./build/cr_client -c -p $APP_PID

# Restore
./build/cr_client -r -p $APP_PID
```

For AMD swap `vGPU-NVIDIA.so` → `vGPU-AMD.so` and add `-m <PARENT_PID>` to
`cr_client` when CRIU needs the original parent.

### 3. Multi-GPU quick start (NVIDIA)

```bash
# Launch any multi-GPU app; the wrapper sets LD_PRELOAD + NCCL_CUMEM_ENABLE
./apps/vllm/launch_multi_gpu.sh \
    python -m vllm.entrypoints.openai.api_server \
        --model /path/to/your/model \
        --tensor-parallel-size 4 \
        --enforce-eager &

# Find the worker PIDs (one per GPU)
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd,)

# Initialize the C/R subsystem in every worker (one-time, after model load)
./build/multi_cr_client -i -p $PIDS

# Checkpoint (frees all GPU memory but keeps processes alive)
./build/multi_cr_client -c -p $PIDS

# … other workloads can now use the GPU …

# Restore (workers resume serving exactly where they paused)
./build/multi_cr_client -r -p $PIDS
```


## VI. Directory Structure

```
GPU-CR/
├── src/                         Core LD_PRELOAD library
│   ├── vGPU.cpp                  signal handlers + C/R loop
│   ├── ipc_hooks.cpp/h           cuMem IPC + peer-access interception
│   ├── ipc_fd_exchange.cpp/h     UDS + SCM_RIGHTS fd exchange
│   ├── nccl_hooks.cpp/h          NCCL communicator tracking
│   ├── common.h                  signals, SHM constants
│   ├── comm/                     SHM control channel
│   ├── backend/                  staging-buffer backends
│   └── GPUs/{NVIDIA,AMD,GPU.h,gpu_factory.cpp}
├── coordinator/
│   ├── cr_client.cpp             single-GPU CLI
│   └── multi_cr_client.cpp       multi-GPU phased orchestrator
├── adapters/
│   ├── nccl/                     NCCL plugin + runtime + 3-file upstream patch
│   └── nvshmem/                  NVSHMEM examples (no patch needed)
├── apps/
│   └── vllm/                     launch + serving scripts for vLLM
├── cuda-checkpoint/              NVIDIA cuda-checkpoint binary (vendored)
└── source/                       README assets (logo, perf charts, gif)
```


## VII. Citation

This project builds on the GCR:

```bibtex
@inproceedings{GCR,
  author    = {Shaoxun Zeng and Tingxu Ren and Jiwu Shu and Youyou Lu},
  title     = {GPU Checkpoint/Restore Made Fast and Lightweight},
  booktitle = {24th USENIX Conference on File and Storage Technologies (FAST'26)},
  year      = {2026},
  address   = {Santa Clara, CA},
  month     = feb,
  publisher = {USENIX Association},
  url       = {https://www.usenix.org/conference/fast26/presentation/zeng}
}
```
