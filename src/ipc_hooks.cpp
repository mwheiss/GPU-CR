/**
 * ipc_hooks.cpp - cuMem IPC Interception via cudaGetDriverEntryPoint Hook
 *
 * NCCL resolves cuMem Driver API functions at runtime through
 * cudaGetDriverEntryPoint() (see nccl-master/src/misc/cudawrap.cc).
 * We hook cudaGetDriverEntryPoint via LD_PRELOAD and, for specific cuMem
 * functions, replace the returned function pointer with our hook.
 *
 * This lets us track all IPC export/import/map operations without modifying
 * NCCL source code. Our own code (nv.cpp) calls cuMem* directly via
 * libcuda.so linking and is NOT affected by this hook.
 *
 * Export-side checkpoint/restore:
 *   The export allocations (cuMemCreate with IPC handle type) retain IPC
 *   metadata in the CUDA driver that prevents cuda-checkpoint from freezing.
 *   We fully tear down exports by saving GPU data to a host buffer, then
 *   cuMemUnmap + cuMemRelease + cuMemAddressFree.  On restore, we re-create
 *   the allocations at the same VA, restore data, and re-export.
 */

#include "ipc_hooks.h"
#include "ipc_fd_exchange.h"
#include "common.h"

#include <cuda.h>
// Prevent the CUDA 11 header's 3-param declaration of cudaGetDriverEntryPoint
// from conflicting with our 4-param hook (needed for CUDA 12 callers).
#define cudaGetDriverEntryPoint cudaGetDriverEntryPoint_cuda11_decl__
#include <cuda_runtime.h>
#undef cudaGetDriverEntryPoint
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <chrono>
#include <unistd.h>
#include <fcntl.h>

// ---------------------------------------------------------------------------
// Internal data structures
// ---------------------------------------------------------------------------

struct IpcImportRecord {
    CUdeviceptr mapped_vaddr;
    size_t size;
    CUmemGenericAllocationHandle handle;
    int access_dev;                          // last device (kept for compat)
    CUmemAccessDesc access_desc;             // last descriptor (kept for compat)
    std::vector<CUmemAccessDesc> all_access; // ALL access descriptors to replay
    bool torn_down;
};

struct IpcExportRecord {
    // Handle originally returned to the application/library.  A rebuild creates
    // a new CUDA handle under the same VA; libraries such as NVSHMEM may still
    // release the original handle during finalize, so we keep it for aliasing.
    CUmemGenericAllocationHandle app_local_handle;
    CUmemGenericAllocationHandle local_handle;
    int exported_fd;
    CUmemAllocationHandleType handle_type;
    size_t size;

    // Export-side VA tracking (for full teardown/rebuild)
    CUdeviceptr mapped_vaddr;
    size_t      mapped_size;
    bool        has_mapping;
    int         access_dev;
    CUmemAccessDesc access_desc;
    bool        has_access;
    bool        torn_down;

    // Saved allocation properties for rebuild (copied from cuMemCreate hook)
    CUmemAllocationProp saved_prop;
    bool has_saved_prop;

    // Host-buffer offset used when the export data was saved. Save may group
    // copies by device, so restore must not assume vector order equals data
    // layout order.
    size_t saved_data_offset;
    bool has_saved_data_offset;
};

/** Record from cuMemCreate hook — tracks allocation properties for rebuild */
struct CuMemCreateRecord {
    size_t size;
    CUmemAllocationProp prop;
};

static std::vector<IpcImportRecord> g_ipc_imports;
static std::vector<IpcExportRecord> g_ipc_exports;

static std::set<CUmemGenericAllocationHandle> g_imported_handles;
static std::map<CUdeviceptr, size_t> g_vaddr_to_import_idx;
static std::map<CUmemGenericAllocationHandle, CuMemCreateRecord> g_created_allocs;
static std::map<CUmemGenericAllocationHandle, CUdeviceptr> g_handle_to_va;
static std::map<CUdeviceptr, size_t> g_vaddr_to_export_idx;
static std::map<CUmemGenericAllocationHandle, CUmemGenericAllocationHandle> g_rebuilt_handle_alias;

// Track non-exported cuMem allocs that need teardown/rebuild for cuda-checkpoint
struct CuMemLocalAllocRecord {
    CUmemGenericAllocationHandle handle;
    CUdeviceptr vaddr;
    size_t size;
    CUmemAllocationProp prop;
    bool torn_down;
};
static std::vector<CuMemLocalAllocRecord> g_local_allocs;  // populated at teardown time

// Track cuMemSetAccess calls for ALL cuMem VAs (not just imports/exports).
// This is needed because NCCL calls cuMemSetAccess BEFORE cuMemExportToShareableHandle,
// so export access descriptors would otherwise be lost.
static std::map<CUdeviceptr, std::vector<CUmemAccessDesc>> g_va_access_descs;

// MUST be recursive: the signal handler (IPC teardown) calls functions that
// lock this mutex, and the signal can fire while the main thread is inside
// a hooked cuMem function that already holds the lock (same thread).
static std::recursive_mutex g_ipc_hook_mutex;
static IpcTimingSnapshot g_timing_snapshot = {};

void ipc_reset_timing_snapshot() {
    std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
    g_timing_snapshot = {};
}

void ipc_get_timing_snapshot(IpcTimingSnapshot* snapshot) {
    if (!snapshot) return;
    std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
    *snapshot = g_timing_snapshot;
}

// ---------------------------------------------------------------------------
// Saved real function pointers
// ---------------------------------------------------------------------------

typedef CUresult (*cuMemCreate_fn)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
typedef CUresult (*cuMemExportToShareableHandle_fn)(void*, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long);
typedef CUresult (*cuMemImportFromShareableHandle_fn)(CUmemGenericAllocationHandle*, void*, CUmemAllocationHandleType);
typedef CUresult (*cuMemMap_fn)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
typedef CUresult (*cuMemUnmap_fn)(CUdeviceptr, size_t);
typedef CUresult (*cuMemSetAccess_fn)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);

static cuMemCreate_fn                   real_cuMemCreate                     = nullptr;
static cuMemExportToShareableHandle_fn   real_cuMemExportToShareableHandle   = nullptr;
static cuMemImportFromShareableHandle_fn real_cuMemImportFromShareableHandle = nullptr;
static cuMemMap_fn                       real_cuMemMap                       = nullptr;
static cuMemUnmap_fn                     real_cuMemUnmap                     = nullptr;
static cuMemSetAccess_fn                 real_cuMemSetAccess                 = nullptr;

typedef CUresult (*cuMemRelease_fn)(CUmemGenericAllocationHandle);
typedef CUresult (*cuMemAddressReserve_fn)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
typedef CUresult (*cuMemAddressFree_fn)(CUdeviceptr, size_t);
typedef CUresult (*cuMemRetainAllocationHandle_fn)(CUmemGenericAllocationHandle*, void*);
typedef CUresult (*cuMemcpyDtoH_fn)(void*, CUdeviceptr, size_t);
typedef CUresult (*cuMemcpyHtoD_fn)(CUdeviceptr, const void*, size_t);

static cuMemRelease_fn                   fn_cuMemRelease                     = nullptr;
static cuMemAddressReserve_fn            fn_cuMemAddressReserve              = nullptr;
static cuMemAddressFree_fn               fn_cuMemAddressFree                 = nullptr;
static cuMemRetainAllocationHandle_fn    fn_cuMemRetainAllocationHandle      = nullptr;
static cuMemcpyDtoH_fn                  fn_cuMemcpyDtoH                     = nullptr;
static cuMemcpyHtoD_fn                  fn_cuMemcpyHtoD                     = nullptr;

typedef CUresult (*cuCtxSetCurrent_fn)(CUcontext);
typedef CUresult (*cuCtxGetCurrent_fn)(CUcontext*);
typedef CUresult (*cuDevicePrimaryCtxRetain_fn)(CUcontext*, CUdevice);
typedef CUresult (*cuDevicePrimaryCtxRelease_fn)(CUdevice);
typedef CUresult (*cuCtxPushCurrent_fn)(CUcontext);
typedef CUresult (*cuCtxPopCurrent_fn)(CUcontext*);
static cuCtxSetCurrent_fn              fn_cuCtxSetCurrent                   = nullptr;
static cuCtxGetCurrent_fn              fn_cuCtxGetCurrent                   = nullptr;
static cuDevicePrimaryCtxRetain_fn     fn_cuDevicePrimaryCtxRetain          = nullptr;
static cuDevicePrimaryCtxRelease_fn    fn_cuDevicePrimaryCtxRelease         = nullptr;
static cuCtxPushCurrent_fn             fn_cuCtxPushCurrent                  = nullptr;
static cuCtxPopCurrent_fn              fn_cuCtxPopCurrent                   = nullptr;

// Async stream functions for batched DtoH/HtoD
typedef CUresult (*cuStreamCreate_fn)(CUstream*, unsigned int);
typedef CUresult (*cuStreamSynchronize_fn)(CUstream);
typedef CUresult (*cuStreamDestroy_fn)(CUstream);
typedef CUresult (*cuMemcpyDtoHAsync_fn)(void*, CUdeviceptr, size_t, CUstream);
typedef CUresult (*cuMemcpyHtoDAsync_fn)(CUdeviceptr, const void*, size_t, CUstream);
typedef CUresult (*cuMemHostRegister_fn)(void*, size_t, unsigned int);
typedef CUresult (*cuMemHostUnregister_fn)(void*);
static cuStreamCreate_fn       fn_cuStreamCreate       = nullptr;
static cuStreamSynchronize_fn  fn_cuStreamSynchronize  = nullptr;
static cuStreamDestroy_fn      fn_cuStreamDestroy      = nullptr;
static cuMemcpyDtoHAsync_fn   fn_cuMemcpyDtoHAsync    = nullptr;
static cuMemcpyHtoDAsync_fn   fn_cuMemcpyHtoDAsync    = nullptr;
static cuMemHostRegister_fn   fn_cuMemHostRegister     = nullptr;
static cuMemHostUnregister_fn fn_cuMemHostUnregister   = nullptr;

typedef cudaError_t (*cudaDeviceEnablePeerAccess_fn)(int, unsigned int);
typedef cudaError_t (*cudaDeviceDisablePeerAccess_fn)(int);
static cudaDeviceEnablePeerAccess_fn real_cudaDeviceEnablePeerAccess = nullptr;
static cudaDeviceDisablePeerAccess_fn real_cudaDeviceDisablePeerAccess = nullptr;
static std::set<int> g_peer_access_devices;
static std::set<int> g_saved_peer_access_devices;
static std::mutex g_peer_access_mutex;

static CUresult CUDAAPI hook_cuMemCreate(
    CUmemGenericAllocationHandle* handle, size_t size,
    const CUmemAllocationProp* prop, unsigned long long flags);
static CUresult CUDAAPI hook_cuMemExportToShareableHandle(
    void* shareableHandle, CUmemGenericAllocationHandle handle,
    CUmemAllocationHandleType handleType, unsigned long long flags);
static CUresult CUDAAPI hook_cuMemImportFromShareableHandle(
    CUmemGenericAllocationHandle* handle, void* osHandle,
    CUmemAllocationHandleType shHandleType);
static CUresult CUDAAPI hook_cuMemMap(
    CUdeviceptr ptr, size_t size, size_t offset,
    CUmemGenericAllocationHandle handle, unsigned long long flags);
static CUresult CUDAAPI hook_cuMemUnmap(CUdeviceptr ptr, size_t size);
static CUresult CUDAAPI hook_cuMemRelease(CUmemGenericAllocationHandle handle);
static CUresult CUDAAPI hook_cuMemSetAccess(
    CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count);

static void resolve_helper_fns() {
    // The preload library exports wrappers for these VMM entry points. Using
    // RTLD_DEFAULT here can resolve back to our own wrapper and recurse until
    // the controller thread segfaults during its first cuMemUnmap/Release.
    if (!fn_cuMemRelease) fn_cuMemRelease = (cuMemRelease_fn)dlsym(RTLD_NEXT, "cuMemRelease");
    if (!fn_cuMemAddressReserve) fn_cuMemAddressReserve = (cuMemAddressReserve_fn)dlsym(RTLD_DEFAULT, "cuMemAddressReserve");
    if (!fn_cuMemAddressFree) fn_cuMemAddressFree = (cuMemAddressFree_fn)dlsym(RTLD_DEFAULT, "cuMemAddressFree");
    if (!fn_cuMemRetainAllocationHandle) fn_cuMemRetainAllocationHandle = (cuMemRetainAllocationHandle_fn)dlsym(RTLD_DEFAULT, "cuMemRetainAllocationHandle");
    if (!fn_cuMemcpyDtoH) fn_cuMemcpyDtoH = (cuMemcpyDtoH_fn)dlsym(RTLD_DEFAULT, "cuMemcpyDtoH_v2");
    if (!fn_cuMemcpyDtoH) fn_cuMemcpyDtoH = (cuMemcpyDtoH_fn)dlsym(RTLD_DEFAULT, "cuMemcpyDtoH");
    if (!fn_cuMemcpyHtoD) fn_cuMemcpyHtoD = (cuMemcpyHtoD_fn)dlsym(RTLD_DEFAULT, "cuMemcpyHtoD_v2");
    if (!fn_cuMemcpyHtoD) fn_cuMemcpyHtoD = (cuMemcpyHtoD_fn)dlsym(RTLD_DEFAULT, "cuMemcpyHtoD");
    if (!real_cuMemCreate) real_cuMemCreate = (cuMemCreate_fn)dlsym(RTLD_NEXT, "cuMemCreate");
    if (!fn_cuCtxSetCurrent) fn_cuCtxSetCurrent = (cuCtxSetCurrent_fn)dlsym(RTLD_DEFAULT, "cuCtxSetCurrent");
    if (!fn_cuCtxGetCurrent) fn_cuCtxGetCurrent = (cuCtxGetCurrent_fn)dlsym(RTLD_DEFAULT, "cuCtxGetCurrent");
    if (!fn_cuDevicePrimaryCtxRetain) fn_cuDevicePrimaryCtxRetain = (cuDevicePrimaryCtxRetain_fn)dlsym(RTLD_DEFAULT, "cuDevicePrimaryCtxRetain");
    if (!fn_cuDevicePrimaryCtxRelease) fn_cuDevicePrimaryCtxRelease = (cuDevicePrimaryCtxRelease_fn)dlsym(RTLD_DEFAULT, "cuDevicePrimaryCtxRelease_v2");
    if (!fn_cuDevicePrimaryCtxRelease) fn_cuDevicePrimaryCtxRelease = (cuDevicePrimaryCtxRelease_fn)dlsym(RTLD_DEFAULT, "cuDevicePrimaryCtxRelease");
    if (!fn_cuCtxPushCurrent) fn_cuCtxPushCurrent = (cuCtxPushCurrent_fn)dlsym(RTLD_DEFAULT, "cuCtxPushCurrent_v2");
    if (!fn_cuCtxPushCurrent) fn_cuCtxPushCurrent = (cuCtxPushCurrent_fn)dlsym(RTLD_DEFAULT, "cuCtxPushCurrent");
    if (!fn_cuCtxPopCurrent) fn_cuCtxPopCurrent = (cuCtxPopCurrent_fn)dlsym(RTLD_DEFAULT, "cuCtxPopCurrent_v2");
    if (!fn_cuCtxPopCurrent) fn_cuCtxPopCurrent = (cuCtxPopCurrent_fn)dlsym(RTLD_DEFAULT, "cuCtxPopCurrent");
    // Async stream functions for batched IPC data transfer
    if (!fn_cuStreamCreate) fn_cuStreamCreate = (cuStreamCreate_fn)dlsym(RTLD_DEFAULT, "cuStreamCreate");
    if (!fn_cuStreamSynchronize) fn_cuStreamSynchronize = (cuStreamSynchronize_fn)dlsym(RTLD_DEFAULT, "cuStreamSynchronize");
    if (!fn_cuStreamDestroy) fn_cuStreamDestroy = (cuStreamDestroy_fn)dlsym(RTLD_DEFAULT, "cuStreamDestroy_v2");
    if (!fn_cuStreamDestroy) fn_cuStreamDestroy = (cuStreamDestroy_fn)dlsym(RTLD_DEFAULT, "cuStreamDestroy");
    if (!fn_cuMemcpyDtoHAsync) fn_cuMemcpyDtoHAsync = (cuMemcpyDtoHAsync_fn)dlsym(RTLD_DEFAULT, "cuMemcpyDtoHAsync_v2");
    if (!fn_cuMemcpyDtoHAsync) fn_cuMemcpyDtoHAsync = (cuMemcpyDtoHAsync_fn)dlsym(RTLD_DEFAULT, "cuMemcpyDtoHAsync");
    if (!fn_cuMemcpyHtoDAsync) fn_cuMemcpyHtoDAsync = (cuMemcpyHtoDAsync_fn)dlsym(RTLD_DEFAULT, "cuMemcpyHtoDAsync_v2");
    if (!fn_cuMemcpyHtoDAsync) fn_cuMemcpyHtoDAsync = (cuMemcpyHtoDAsync_fn)dlsym(RTLD_DEFAULT, "cuMemcpyHtoDAsync");
    if (!fn_cuMemHostRegister) fn_cuMemHostRegister = (cuMemHostRegister_fn)dlsym(RTLD_DEFAULT, "cuMemHostRegister_v2");
    if (!fn_cuMemHostRegister) fn_cuMemHostRegister = (cuMemHostRegister_fn)dlsym(RTLD_DEFAULT, "cuMemHostRegister");
    if (!fn_cuMemHostUnregister) fn_cuMemHostUnregister = (cuMemHostUnregister_fn)dlsym(RTLD_DEFAULT, "cuMemHostUnregister");
}

// ---------------------------------------------------------------------------
// Hook implementations
// ---------------------------------------------------------------------------

extern "C" cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    if (!real_cudaDeviceEnablePeerAccess) {
        real_cudaDeviceEnablePeerAccess =
            (cudaDeviceEnablePeerAccess_fn)dlsym(RTLD_NEXT, "cudaDeviceEnablePeerAccess");
    }

    if (!real_cudaDeviceEnablePeerAccess) {
        fprintf(stderr, "[IPC-HOOK] WARNING: real cudaDeviceEnablePeerAccess not found\n");
        return cudaSuccess;
    }

    cudaError_t err = real_cudaDeviceEnablePeerAccess(peerDevice, flags);
    if (err == cudaSuccess) {
        std::lock_guard<std::mutex> lock(g_peer_access_mutex);
        g_peer_access_devices.insert(peerDevice);
        fprintf(stderr, "[IPC-HOOK] P2P access enabled to device %d (tracked=%zu)\n",
                peerDevice, g_peer_access_devices.size());
    } else if (err == cudaErrorPeerAccessAlreadyEnabled) {
        std::lock_guard<std::mutex> lock(g_peer_access_mutex);
        g_peer_access_devices.insert(peerDevice);
        cudaGetLastError();
        fprintf(stderr, "[IPC-HOOK] P2P access already enabled to device %d (tracked=%zu)\n",
                peerDevice, g_peer_access_devices.size());
        err = cudaSuccess;
    }
    return err;
}

extern "C" cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) {
    if (!real_cudaDeviceDisablePeerAccess) {
        real_cudaDeviceDisablePeerAccess =
            (cudaDeviceDisablePeerAccess_fn)dlsym(RTLD_NEXT, "cudaDeviceDisablePeerAccess");
    }

    if (!real_cudaDeviceDisablePeerAccess) return cudaSuccess;

    cudaError_t err = real_cudaDeviceDisablePeerAccess(peerDevice);
    if (err == cudaSuccess) {
        std::lock_guard<std::mutex> lock(g_peer_access_mutex);
        g_peer_access_devices.erase(peerDevice);
        fprintf(stderr, "[IPC-HOOK] P2P access disabled to device %d\n", peerDevice);
    }
    return err;
}

int ipc_disable_all_peer_access() {
    if (!real_cudaDeviceDisablePeerAccess) {
        real_cudaDeviceDisablePeerAccess =
            (cudaDeviceDisablePeerAccess_fn)dlsym(RTLD_NEXT, "cudaDeviceDisablePeerAccess");
    }
    if (!real_cudaDeviceDisablePeerAccess) return 0;

    std::lock_guard<std::mutex> lock(g_peer_access_mutex);
    g_saved_peer_access_devices = g_peer_access_devices;

    int disabled = 0;
    for (int dev : g_peer_access_devices) {
        cudaError_t err = real_cudaDeviceDisablePeerAccess(dev);
        if (err == cudaSuccess || err == cudaErrorPeerAccessNotEnabled) {
            disabled++;
        }
        cudaGetLastError();
    }

    fprintf(stderr, "[IPC-HOOK] Pre-checkpoint: disabled %d/%zu P2P peer entries\n",
            disabled, g_peer_access_devices.size());
    g_peer_access_devices.clear();
    return disabled;
}

int ipc_reenable_all_peer_access() {
    if (!real_cudaDeviceEnablePeerAccess) {
        real_cudaDeviceEnablePeerAccess =
            (cudaDeviceEnablePeerAccess_fn)dlsym(RTLD_NEXT, "cudaDeviceEnablePeerAccess");
    }
    if (!real_cudaDeviceEnablePeerAccess) return 0;

    std::lock_guard<std::mutex> lock(g_peer_access_mutex);

    int enabled = 0;
    for (int dev : g_saved_peer_access_devices) {
        cudaError_t err = real_cudaDeviceEnablePeerAccess(dev, 0);
        if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled) {
            g_peer_access_devices.insert(dev);
            enabled++;
        }
        cudaGetLastError();
    }

    fprintf(stderr, "[IPC-HOOK] Post-restore: re-enabled %d/%zu P2P peer entries\n",
            enabled, g_saved_peer_access_devices.size());
    g_saved_peer_access_devices.clear();
    return enabled;
}

static CUresult CUDAAPI hook_cuMemCreate(
    CUmemGenericAllocationHandle* handle, size_t size,
    const CUmemAllocationProp* prop, unsigned long long flags)
{
    gpu_cr_ensure_control_thread();
    if (!real_cuMemCreate) {
        fprintf(stderr, "[IPC-HOOK] ERROR: real cuMemCreate not available\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    CUresult res = real_cuMemCreate(handle, size, prop, flags);
    if (res == CUDA_SUCCESS && handle && prop) {
        if (prop->requestedHandleTypes != 0) {
            std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
            CuMemCreateRecord rec;
            rec.size = size;
            rec.prop = *prop;
            g_created_allocs[*handle] = rec;
            fprintf(stderr, "[IPC-HOOK] cuMemCreate: handle=0x%llx size=%zu dev=%d handleTypes=0x%x\n",
                    (unsigned long long)*handle, size, prop->location.id,
                    (unsigned)prop->requestedHandleTypes);
        }
    }
    return res;
}

static CUresult CUDAAPI hook_cuMemExportToShareableHandle(
    void* shareableHandle, CUmemGenericAllocationHandle handle,
    CUmemAllocationHandleType handleType, unsigned long long flags)
{
    if (!real_cuMemExportToShareableHandle) {
        fprintf(stderr, "[IPC-HOOK] ERROR: real cuMemExportToShareableHandle not available\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    CUresult res = real_cuMemExportToShareableHandle(shareableHandle, handle, handleType, flags);
    if (res == CUDA_SUCCESS) {
        std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);

        IpcExportRecord rec = {};
        rec.app_local_handle = handle;
        rec.local_handle = handle;
        rec.handle_type = handleType;
        rec.exported_fd = -1;
        rec.has_mapping = false;
        rec.has_access = false;
        rec.torn_down = false;
        rec.mapped_vaddr = 0;
        rec.mapped_size = 0;
        rec.access_dev = -1;
        rec.has_saved_prop = false;
        memset(&rec.access_desc, 0, sizeof(rec.access_desc));
        memset(&rec.saved_prop, 0, sizeof(rec.saved_prop));

        // Save allocation properties from cuMemCreate tracking
        auto create_it = g_created_allocs.find(handle);
        if (create_it != g_created_allocs.end()) {
            rec.saved_prop = create_it->second.prop;
            rec.has_saved_prop = true;
        }

        if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
            rec.exported_fd = *(int*)shareableHandle;
        }

        auto it = g_created_allocs.find(handle);
        if (it != g_created_allocs.end()) {
            rec.size = it->second.size;
        }

        auto va_it = g_handle_to_va.find(handle);
        if (va_it != g_handle_to_va.end()) {
            rec.mapped_vaddr = va_it->second;
            rec.mapped_size = rec.size;
            rec.has_mapping = true;
        }

        size_t idx = g_ipc_exports.size();
        g_ipc_exports.push_back(rec);

        if (rec.has_mapping) {
            g_vaddr_to_export_idx[rec.mapped_vaddr] = idx;
        }

        fprintf(stderr, "[IPC-HOOK] Export: handle=0x%llx type=%d fd=%d size=%zu vaddr=%p (total=%zu)\n",
                (unsigned long long)handle, (int)handleType, rec.exported_fd,
                rec.size, (void*)rec.mapped_vaddr, g_ipc_exports.size());
    }
    return res;
}

static CUresult CUDAAPI hook_cuMemImportFromShareableHandle(
    CUmemGenericAllocationHandle* handle, void* osHandle,
    CUmemAllocationHandleType shHandleType)
{
    if (!real_cuMemImportFromShareableHandle) {
        fprintf(stderr, "[IPC-HOOK] ERROR: real cuMemImportFromShareableHandle not available\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    CUresult res = real_cuMemImportFromShareableHandle(handle, osHandle, shHandleType);
    if (res == CUDA_SUCCESS && handle) {
        std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
        g_imported_handles.insert(*handle);
        fprintf(stderr, "[IPC-HOOK] Import: handle=0x%llx type=%d (tracked=%zu)\n",
                (unsigned long long)*handle, (int)shHandleType, g_imported_handles.size());
    }
    return res;
}

static CUresult CUDAAPI hook_cuMemMap(
    CUdeviceptr ptr, size_t size, size_t offset,
    CUmemGenericAllocationHandle handle, unsigned long long flags)
{
    if (!real_cuMemMap) {
        fprintf(stderr, "[IPC-HOOK] ERROR: real cuMemMap not available\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    CUresult res = real_cuMemMap(ptr, size, offset, handle, flags);
    if (res == CUDA_SUCCESS) {
        std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);

        if (g_imported_handles.count(handle)) {
            IpcImportRecord rec = {};
            rec.mapped_vaddr = ptr;
            rec.size = size;
            rec.handle = handle;
            rec.access_dev = -1;
            memset(&rec.access_desc, 0, sizeof(rec.access_desc));
            rec.torn_down = false;

            size_t idx = g_ipc_imports.size();
            g_ipc_imports.push_back(rec);
            g_vaddr_to_import_idx[ptr] = idx;

            fprintf(stderr, "[IPC-HOOK] Map IPC import: vaddr=%p size=%zu handle=0x%llx (total=%zu)\n",
                    (void*)ptr, size, (unsigned long long)handle, g_ipc_imports.size());
        }

        if (g_created_allocs.count(handle)) {
            g_handle_to_va[handle] = ptr;
            fprintf(stderr, "[IPC-HOOK] Map local IPC-capable alloc: vaddr=%p size=%zu handle=0x%llx\n",
                    (void*)ptr, size, (unsigned long long)handle);

            for (size_t i = 0; i < g_ipc_exports.size(); i++) {
                auto& exp = g_ipc_exports[i];
                if (exp.local_handle == handle && !exp.has_mapping) {
                    exp.mapped_vaddr = ptr;
                    exp.mapped_size = size;
                    exp.has_mapping = true;
                    g_vaddr_to_export_idx[ptr] = i;
                }
            }
        }
    }
    return res;
}

static CUresult CUDAAPI hook_cuMemUnmap(CUdeviceptr ptr, size_t size) {
    if (!real_cuMemUnmap) {
        fprintf(stderr, "[IPC-HOOK] ERROR: real cuMemUnmap not available\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    CUresult res = real_cuMemUnmap(ptr, size);
    if (res == CUDA_SUCCESS) {
        std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
        auto it = g_vaddr_to_import_idx.find(ptr);
        if (it != g_vaddr_to_import_idx.end()) {
            size_t idx = it->second;
            if (idx < g_ipc_imports.size() && !g_ipc_imports[idx].torn_down) {
                fprintf(stderr, "[IPC-HOOK] Unmap IPC import (normal): vaddr=%p size=%zu\n",
                        (void*)ptr, size);
                g_ipc_imports[idx].torn_down = true;
            }
        }
    }
    return res;
}

static CUresult CUDAAPI hook_cuMemRelease(CUmemGenericAllocationHandle handle) {
    if (!fn_cuMemRelease) resolve_helper_fns();
    if (!fn_cuMemRelease) {
        fprintf(stderr, "[IPC-HOOK] ERROR: real cuMemRelease not available\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    CUmemGenericAllocationHandle actual_handle = handle;
    bool alias_release = false;
    {
        std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
        auto alias_it = g_rebuilt_handle_alias.find(handle);
        if (alias_it != g_rebuilt_handle_alias.end()) {
            actual_handle = alias_it->second;
            alias_release = true;
            fprintf(stderr, "[IPC-HOOK] Release alias cuMem handle app=0x%llx actual=0x%llx\n",
                    (unsigned long long)handle, (unsigned long long)actual_handle);
        }
    }

    CUresult res = fn_cuMemRelease(actual_handle);
    if (res == CUDA_SUCCESS) {
        std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
        auto cleanup_handle = [&](CUmemGenericAllocationHandle h) {
            auto va_it = g_handle_to_va.find(h);
            if (va_it != g_handle_to_va.end()) {
                CUdeviceptr va = va_it->second;
                g_va_access_descs.erase(va);
                g_vaddr_to_export_idx.erase(va);
                g_handle_to_va.erase(va_it);
            }
            g_created_allocs.erase(h);
            g_imported_handles.erase(h);
            for (auto& rec : g_ipc_imports) {
                if (rec.handle == h) rec.handle = 0;
            }
            for (auto& rec : g_ipc_exports) {
                if (rec.local_handle == h) rec.local_handle = 0;
            }
            for (auto& rec : g_local_allocs) {
                if (rec.handle == h) rec.torn_down = true;
            }
        };

        cleanup_handle(handle);
        if (actual_handle != handle) cleanup_handle(actual_handle);
        for (auto it = g_rebuilt_handle_alias.begin(); it != g_rebuilt_handle_alias.end();) {
            if (it->first == handle || it->first == actual_handle ||
                it->second == handle || it->second == actual_handle) {
                it = g_rebuilt_handle_alias.erase(it);
            } else {
                ++it;
            }
        }
        fprintf(stderr, "[IPC-HOOK] Release tracked cuMem handle=0x%llx\n",
                (unsigned long long)(alias_release ? actual_handle : handle));
    }
    return res;
}

static CUresult CUDAAPI hook_cuMemSetAccess(
    CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count)
{
    if (!real_cuMemSetAccess) {
        fprintf(stderr, "[IPC-HOOK] ERROR: real cuMemSetAccess not available\n");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    CUresult res = real_cuMemSetAccess(ptr, size, desc, count);
    if (res == CUDA_SUCCESS && desc && count > 0) {
        std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);

        // Track access for ALL cuMem VAs — needed because NCCL may call
        // cuMemSetAccess BEFORE cuMemExportToShareableHandle, so export
        // access would otherwise be lost.
        auto& va_descs = g_va_access_descs[ptr];
        for (size_t d = 0; d < count; d++) {
            bool found = false;
            for (auto& existing : va_descs) {
                if (existing.location.type == desc[d].location.type &&
                    existing.location.id == desc[d].location.id) {
                    existing.flags = desc[d].flags;
                    found = true;
                    break;
                }
            }
            if (!found) {
                va_descs.push_back(desc[d]);
            }
        }

        auto it = g_vaddr_to_import_idx.find(ptr);
        if (it != g_vaddr_to_import_idx.end()) {
            size_t idx = it->second;
            if (idx < g_ipc_imports.size()) {
                g_ipc_imports[idx].access_desc = desc[0];
                g_ipc_imports[idx].access_dev = desc[0].location.id;
                // Accumulate ALL access descriptors for replay during rebuild
                for (size_t d = 0; d < count; d++) {
                    // Avoid duplicates: check if this device already recorded
                    bool found = false;
                    for (auto& existing : g_ipc_imports[idx].all_access) {
                        if (existing.location.type == desc[d].location.type &&
                            existing.location.id == desc[d].location.id) {
                            existing.flags = desc[d].flags; // update flags
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        g_ipc_imports[idx].all_access.push_back(desc[d]);
                    }
                }
                fprintf(stderr, "[IPC-HOOK] SetAccess on IPC import: vaddr=%p dev=%d (total_access=%zu)\n",
                        (void*)ptr, desc[0].location.id, g_ipc_imports[idx].all_access.size());
            }
        }

        auto eit = g_vaddr_to_export_idx.find(ptr);
        if (eit != g_vaddr_to_export_idx.end()) {
            size_t idx = eit->second;
            if (idx < g_ipc_exports.size()) {
                g_ipc_exports[idx].access_desc = desc[0];
                g_ipc_exports[idx].access_dev = desc[0].location.id;
                g_ipc_exports[idx].has_access = true;
                fprintf(stderr, "[IPC-HOOK] SetAccess on IPC export: vaddr=%p dev=%d\n",
                        (void*)ptr, desc[0].location.id);
            }
        }
    }
    return res;
}

// ---------------------------------------------------------------------------
// cudaGetDriverEntryPoint hook
// ---------------------------------------------------------------------------

struct HookEntry {
    const char* symbol;
    void* hook_fn;
    void** real_fn_storage;
};

static HookEntry g_hook_table[] = {
    {"cuMemCreate",                    (void*)hook_cuMemCreate,                    (void**)&real_cuMemCreate},
    {"cuMemExportToShareableHandle",   (void*)hook_cuMemExportToShareableHandle,   (void**)&real_cuMemExportToShareableHandle},
    {"cuMemImportFromShareableHandle", (void*)hook_cuMemImportFromShareableHandle, (void**)&real_cuMemImportFromShareableHandle},
    {"cuMemMap",                       (void*)hook_cuMemMap,                       (void**)&real_cuMemMap},
    {"cuMemUnmap",                     (void*)hook_cuMemUnmap,                     (void**)&real_cuMemUnmap},
    {"cuMemRelease",                   (void*)hook_cuMemRelease,                   (void**)&fn_cuMemRelease},
    {"cuMemSetAccess",                 (void*)hook_cuMemSetAccess,                 (void**)&real_cuMemSetAccess},
    {nullptr, nullptr, nullptr}
};

static void try_intercept(const char* symbol, void** funcPtr) {
    if (!symbol || !funcPtr || !*funcPtr) return;
    for (int i = 0; g_hook_table[i].symbol != nullptr; i++) {
        if (strcmp(symbol, g_hook_table[i].symbol) == 0) {
            *(g_hook_table[i].real_fn_storage) = *funcPtr;
            fprintf(stderr, "[IPC-HOOK] Intercepted %s (real=%p)\n", symbol, *funcPtr);
            *funcPtr = g_hook_table[i].hook_fn;
            return;
        }
    }
}

extern "C" void* dlvsym(void* handle, const char* symbol, const char* version);
#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
extern "C" CUresult cuGetProcAddress(const char* symbol, void** pfn,
                                     int cudaVersion, cuuint64_t flags);
extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* symbol, void** pfn,
                                                int cudaVersion, cuuint64_t flags,
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
                                                CUdriverProcAddressQueryResult* symbolStatus);
#else
                                                void* symbolStatus);
#endif
extern "C" cudaError_t cudaGetDriverEntryPoint(
    const char* symbol, void** funcPtr, unsigned long long flags,
    enum cudaDriverEntryPointQueryResult* driverStatus);
extern "C" cudaError_t cudaGetDriverEntryPointByVersion(
    const char* symbol, void** funcPtr, unsigned int cudaVersion,
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12060
    unsigned long long flags, cudaDriverEntryPointQueryResult* driverStatus);
#else
    unsigned long long flags, void* driverStatus);
#endif

typedef void* (*real_dlsym_fn)(void*, const char*);
static thread_local bool g_inside_dlsym_hook = false;

extern "C" void* dlsym(void* handle, const char* symbol) {
    static real_dlsym_fn real_dlsym =
        (real_dlsym_fn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    if (!real_dlsym) return nullptr;
    static int trace_count = 0;
    const char* trace = getenv("GCR_TRACE_DLSYM");
    if (trace && trace[0] != '\0' && trace[0] != '0' && trace_count < 300) {
        fprintf(stderr, "[IPC-HOOK] dlsym trace #%d handle=%p symbol=%s\n",
                trace_count, handle, symbol ? symbol : "(null)");
        trace_count++;
    }

    if (!g_inside_dlsym_hook && handle != RTLD_NEXT && symbol) {
        if (strcmp(symbol, "cuGetProcAddress") == 0) {
            fprintf(stderr, "[IPC-HOOK] dlsym intercepted cuGetProcAddress (handle=%p)\n", handle);
            return (void*)cuGetProcAddress;
        }
        if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
            fprintf(stderr, "[IPC-HOOK] dlsym intercepted cuGetProcAddress_v2 (handle=%p)\n", handle);
            return (void*)cuGetProcAddress_v2;
        }
        if (strcmp(symbol, "cudaGetDriverEntryPoint") == 0) {
            fprintf(stderr, "[IPC-HOOK] dlsym intercepted cudaGetDriverEntryPoint (handle=%p)\n", handle);
            return (void*)cudaGetDriverEntryPoint;
        }
        if (strcmp(symbol, "cudaGetDriverEntryPointByVersion") == 0) {
            fprintf(stderr, "[IPC-HOOK] dlsym intercepted cudaGetDriverEntryPointByVersion (handle=%p)\n", handle);
            return (void*)cudaGetDriverEntryPointByVersion;
        }
        for (int i = 0; g_hook_table[i].symbol != nullptr; i++) {
            if (strcmp(symbol, g_hook_table[i].symbol) == 0) {
                g_inside_dlsym_hook = true;
                // Always resolve the implementation after this preload
                // library. RTLD_DEFAULT would select our exported wrapper for
                // symbols such as cuMemUnmap, turning the saved "real"
                // function pointer into a recursive self-call.
                void* real_symbol = real_dlsym(RTLD_NEXT, symbol);
                g_inside_dlsym_hook = false;
                if (real_symbol) {
                    *(g_hook_table[i].real_fn_storage) = real_symbol;
                    fprintf(stderr, "[IPC-HOOK] dlsym intercepted %s (real=%p handle=%p)\n",
                            symbol, real_symbol, handle);
                    return g_hook_table[i].hook_fn;
                }
            }
        }
    }

    g_inside_dlsym_hook = true;
    void* ret = real_dlsym(handle, symbol);
    g_inside_dlsym_hook = false;
    return ret;
}

// ---------------------------------------------------------------------------
// Hook function pointer types.
//
// CUDA 12 added extra status-output parameters to cudaGetDriverEntryPoint and
// cuGetProcAddress, AND renamed cuGetProcAddress → cuGetProcAddress_v2 via a
// macro.  We compile against whatever headers are available, but at RUNTIME
// the NCCL library may call either the old or the new symbol.  To guarantee
// interception regardless of compile-time vs. run-time CUDA version we:
//   • Always provide hooks for BOTH symbol names (e.g. cuGetProcAddress AND
//     cuGetProcAddress_v2) using void* for the optional status parameters.
//   • Use dlsym(RTLD_NEXT, ...) to find the real implementation at runtime.
// ---------------------------------------------------------------------------

// 3-param cudaGetDriverEntryPoint (CUDA 11 & CUDA 12 C API)
typedef cudaError_t (*cudaGetDriverEntryPoint_3p_fn)(
    const char*, void**, unsigned long long);
// 4-param cudaGetDriverEntryPoint (CUDA 12 C++ overload)
typedef cudaError_t (*cudaGetDriverEntryPoint_4p_fn)(
    const char*, void**, unsigned long long, enum cudaDriverEntryPointQueryResult*);
// cudaGetDriverEntryPointByVersion (CUDA 12.6+ only)
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12060
typedef cudaError_t (*cudaGetDriverEntryPointByVersion_fn)(
    const char*, void**, unsigned int, unsigned long long, cudaDriverEntryPointQueryResult*);
#else
typedef cudaError_t (*cudaGetDriverEntryPointByVersion_fn)(
    const char*, void**, unsigned int, unsigned long long, void*);
#endif
// cuGetProcAddress — 4-param (CUDA 11) and 5-param (CUDA 12 / _v2)
typedef CUresult (*cuGetProcAddress_4p_fn)(const char*, void**, int, cuuint64_t);
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
typedef CUresult (*cuGetProcAddress_5p_fn)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
#else
typedef CUresult (*cuGetProcAddress_5p_fn)(const char*, void**, int, cuuint64_t, void*);
#endif

// Prevent the CUDA header macro from renaming our hook function.
// In CUDA 12 headers, cuGetProcAddress is a macro for cuGetProcAddress_v2.
// We #undef it so we can define both symbols explicitly.
#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif

static int g_hook_call_count_4param = 0;
static int g_hook_call_count_byversion = 0;
static int g_hook_call_count_procaddr = 0;

extern "C" {

// ---------------------------------------------------------------------------
// Detection hooks for old-style CUDA IPC (cudaIpc* API)
// ---------------------------------------------------------------------------
static int g_cudaIpc_warning_count = 0;

cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) {
    if (g_cudaIpc_warning_count++ < 5) {
        fprintf(stderr, "[IPC-HOOK] *** WARNING: cudaIpcGetMemHandle called! devPtr=%p ***\n", devPtr);
    }
    typedef cudaError_t (*fn_t)(cudaIpcMemHandle_t*, void*);
    static fn_t real_fn = (fn_t)dlsym(RTLD_NEXT, "cudaIpcGetMemHandle");
    if (!real_fn) return cudaErrorNotSupported;
    return real_fn(handle, devPtr);
}

cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) {
    if (g_cudaIpc_warning_count++ < 5) {
        fprintf(stderr, "[IPC-HOOK] *** WARNING: cudaIpcOpenMemHandle called! flags=%u ***\n", flags);
    }
    typedef cudaError_t (*fn_t)(void**, cudaIpcMemHandle_t, unsigned int);
    static fn_t real_fn = (fn_t)dlsym(RTLD_NEXT, "cudaIpcOpenMemHandle");
    if (!real_fn) return cudaErrorNotSupported;
    return real_fn(devPtr, handle, flags);
}

cudaError_t cudaIpcCloseMemHandle(void* devPtr) {
    fprintf(stderr, "[IPC-HOOK] *** cudaIpcCloseMemHandle called! devPtr=%p ***\n", devPtr);
    typedef cudaError_t (*fn_t)(void*);
    static fn_t real_fn = (fn_t)dlsym(RTLD_NEXT, "cudaIpcCloseMemHandle");
    if (!real_fn) return cudaErrorNotSupported;
    return real_fn(devPtr);
}

CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size,
                             const CUmemAllocationProp* prop,
                             unsigned long long flags) {
    if (!real_cuMemCreate)
        real_cuMemCreate = (cuMemCreate_fn)dlsym(RTLD_NEXT, "cuMemCreate");
    return hook_cuMemCreate(handle, size, prop, flags);
}

CUresult CUDAAPI cuMemExportToShareableHandle(void* shareableHandle,
                                              CUmemGenericAllocationHandle handle,
                                              CUmemAllocationHandleType handleType,
                                              unsigned long long flags) {
    if (!real_cuMemExportToShareableHandle)
        real_cuMemExportToShareableHandle =
            (cuMemExportToShareableHandle_fn)dlsym(RTLD_NEXT, "cuMemExportToShareableHandle");
    return hook_cuMemExportToShareableHandle(shareableHandle, handle, handleType, flags);
}

CUresult CUDAAPI cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle,
                                                void* osHandle,
                                                CUmemAllocationHandleType shHandleType) {
    if (!real_cuMemImportFromShareableHandle)
        real_cuMemImportFromShareableHandle =
            (cuMemImportFromShareableHandle_fn)dlsym(RTLD_NEXT, "cuMemImportFromShareableHandle");
    return hook_cuMemImportFromShareableHandle(handle, osHandle, shHandleType);
}

CUresult CUDAAPI cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                          CUmemGenericAllocationHandle handle,
                          unsigned long long flags) {
    if (!real_cuMemMap)
        real_cuMemMap = (cuMemMap_fn)dlsym(RTLD_NEXT, "cuMemMap");
    return hook_cuMemMap(ptr, size, offset, handle, flags);
}

CUresult CUDAAPI cuMemUnmap(CUdeviceptr ptr, size_t size) {
    if (!real_cuMemUnmap)
        real_cuMemUnmap = (cuMemUnmap_fn)dlsym(RTLD_NEXT, "cuMemUnmap");
    return hook_cuMemUnmap(ptr, size);
}

CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle) {
    if (!fn_cuMemRelease)
        fn_cuMemRelease = (cuMemRelease_fn)dlsym(RTLD_NEXT, "cuMemRelease");
    return hook_cuMemRelease(handle);
}

CUresult CUDAAPI cuMemSetAccess(CUdeviceptr ptr, size_t size,
                                const CUmemAccessDesc* desc, size_t count) {
    if (!real_cuMemSetAccess)
        real_cuMemSetAccess = (cuMemSetAccess_fn)dlsym(RTLD_NEXT, "cuMemSetAccess");
    return hook_cuMemSetAccess(ptr, size, desc, count);
}

// ---------------------------------------------------------------------------
// IPC event hooks
// ---------------------------------------------------------------------------
struct IpcEventRecord {
    cudaEvent_t event;
    bool is_opened;
    bool torn_down;
};

static std::vector<IpcEventRecord> g_ipc_events;
static std::mutex g_ipc_events_mutex;

cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) {
    fprintf(stderr, "[IPC-HOOK] cudaIpcGetEventHandle: event=%p\n", (void*)event);
    typedef cudaError_t (*fn_t)(cudaIpcEventHandle_t*, cudaEvent_t);
    static fn_t real_fn = (fn_t)dlsym(RTLD_NEXT, "cudaIpcGetEventHandle");
    if (!real_fn) return cudaErrorNotSupported;
    cudaError_t ret = real_fn(handle, event);
    if (ret == cudaSuccess) {
        std::lock_guard<std::mutex> lock(g_ipc_events_mutex);
        IpcEventRecord rec = {event, false, false};
        g_ipc_events.push_back(rec);
    }
    return ret;
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle) {
    fprintf(stderr, "[IPC-HOOK] cudaIpcOpenEventHandle called\n");
    typedef cudaError_t (*fn_t)(cudaEvent_t*, cudaIpcEventHandle_t);
    static fn_t real_fn = (fn_t)dlsym(RTLD_NEXT, "cudaIpcOpenEventHandle");
    if (!real_fn) return cudaErrorNotSupported;
    cudaError_t ret = real_fn(event, handle);
    if (ret == cudaSuccess && event && *event) {
        std::lock_guard<std::mutex> lock(g_ipc_events_mutex);
        IpcEventRecord rec = {*event, true, false};
        g_ipc_events.push_back(rec);
    }
    return ret;
}

// ---------------------------------------------------------------------------
// cudaGetDriverEntryPoint hook — 4-param (handles both CUDA 11 and 12 callers)
//
// CUDA 11: cudaGetDriverEntryPoint(symbol, funcPtr, flags)           — 3 params
// CUDA 12: cudaGetDriverEntryPoint(symbol, funcPtr, flags, status)   — 4 params
//
// We define the 4-param version.  CUDA 12 callers pass a non-NULL
// driverStatus pointer and CHECK the value — if we don't fill it in,
// the caller sees garbage and treats the lookup as failed.
//
// CUDA 11 callers (3-arg) leave rcx uninitialized on x86-64, but all
// callers in this system are compiled with CUDA 12, so this is safe.
//
// ---------------------------------------------------------------------------

cudaError_t cudaGetDriverEntryPoint(
    const char* symbol, void** funcPtr, unsigned long long flags,
    enum cudaDriverEntryPointQueryResult* driverStatus)
{
    static cudaGetDriverEntryPoint_4p_fn real_fn = nullptr;
    if (!real_fn) {
        real_fn = (cudaGetDriverEntryPoint_4p_fn)dlsym(RTLD_NEXT, "cudaGetDriverEntryPoint");
        if (!real_fn) return cudaErrorNotSupported;
        fprintf(stderr, "[IPC-HOOK] cudaGetDriverEntryPoint hook activated (4-param)\n");
    }
    g_hook_call_count_4param++;
    if (g_hook_call_count_4param <= 10 || (symbol && strncmp(symbol, "cuMem", 5) == 0)) {
        fprintf(stderr, "[IPC-HOOK] cudaGetDriverEntryPoint(#%d): symbol=%s\n",
                g_hook_call_count_4param, symbol ? symbol : "(null)");
    }
    // Real cudart's cudaGetDriverEntryPoint is 4-param in CUDA 12.5+
    // (verified by disassembly of libcudart.so.12 from CUDA 12.9 — reads rcx
    //  as driverStatus). Calling it as 3-param leaves rcx with garbage and
    //  the real impl derefs it → segfault. Pass driverStatus straight
    //  through; the real function writes it. (Extra arg is harmless on
    //  the older 3-param impl.)
    cudaError_t ret = real_fn(symbol, funcPtr, flags, driverStatus);

    if (ret == cudaSuccess && funcPtr && *funcPtr) {
        try_intercept(symbol, funcPtr);
    }
    return ret;
}

cudaError_t cudaGetDriverEntryPointByVersion(
    const char* symbol, void** funcPtr, unsigned int cudaVersion,
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12060
    unsigned long long flags, cudaDriverEntryPointQueryResult* driverStatus)
#else
    unsigned long long flags, void* driverStatus)
#endif
{
    static cudaGetDriverEntryPointByVersion_fn real_fn = nullptr;
    if (!real_fn) {
        real_fn = (cudaGetDriverEntryPointByVersion_fn)dlsym(RTLD_NEXT, "cudaGetDriverEntryPointByVersion");
        if (!real_fn) return cudaErrorNotSupported;
        fprintf(stderr, "[IPC-HOOK] cudaGetDriverEntryPointByVersion hook activated\n");
    }
    g_hook_call_count_byversion++;
    if (g_hook_call_count_byversion <= 10 || (symbol && strncmp(symbol, "cuMem", 5) == 0)) {
        fprintf(stderr, "[IPC-HOOK] cudaGetDriverEntryPointByVersion(#%d): symbol=%s version=%u\n",
                g_hook_call_count_byversion, symbol ? symbol : "(null)", cudaVersion);
    }

    cudaError_t ret = real_fn(symbol, funcPtr, cudaVersion, flags, driverStatus);
    if (ret == cudaSuccess && funcPtr && *funcPtr) {
        try_intercept(symbol, funcPtr);
    }
    return ret;
}

// ---------------------------------------------------------------------------
// cuGetProcAddress hooks — provide BOTH symbol names so that the correct one
// is intercepted regardless of compile-time vs. run-time CUDA version.
//
//   CUDA 11: runtime calls  cuGetProcAddress       (4 params)
//   CUDA 12: runtime calls  cuGetProcAddress_v2    (5 params)
//
// We define both with C linkage and match the CUDA 12 status parameter type.
// ---------------------------------------------------------------------------

// Helper: shared logic for cuGetProcAddress interception
static CUresult cuGetProcAddress_common(const char* symbol, void** pfn,
                                         const char* hook_label) {
    g_hook_call_count_procaddr++;
    if (g_hook_call_count_procaddr <= 10 || (symbol && strncmp(symbol, "cuMem", 5) == 0)) {
        fprintf(stderr, "[IPC-HOOK] %s(#%d): symbol=%s\n",
                hook_label, g_hook_call_count_procaddr, symbol ? symbol : "(null)");
    }
    if (pfn && *pfn) {
        try_intercept(symbol, pfn);
    }
    return CUDA_SUCCESS; // not used — caller returns real result
}

// cuGetProcAddress — 4 param (CUDA 11 symbol)
CUresult cuGetProcAddress(const char* symbol, void** pfn,
                          int cudaVersion, cuuint64_t flags)
{
    static cuGetProcAddress_4p_fn real_fn = nullptr;
    if (!real_fn) {
        // Try the exact symbol first, then fall back
        real_fn = (cuGetProcAddress_4p_fn)dlsym(RTLD_NEXT, "cuGetProcAddress");
        if (!real_fn) return CUDA_ERROR_NOT_FOUND;
        fprintf(stderr, "[IPC-HOOK] cuGetProcAddress hook activated (4-param)\n");
    }
    CUresult ret = real_fn(symbol, pfn, cudaVersion, flags);
    if (ret == CUDA_SUCCESS) cuGetProcAddress_common(symbol, pfn, "cuGetProcAddress");
    return ret;
}

// cuGetProcAddress_v2 — 5 param (CUDA 12+ symbol)
CUresult CUDAAPI cuGetProcAddress_v2(const char* symbol, void** pfn,
                                     int cudaVersion, cuuint64_t flags,
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
                                     CUdriverProcAddressQueryResult* symbolStatus)
#else
                                     void* symbolStatus)
#endif
{
    static cuGetProcAddress_5p_fn real_fn = nullptr;
    if (!real_fn) {
        real_fn = (cuGetProcAddress_5p_fn)dlsym(RTLD_NEXT, "cuGetProcAddress_v2");
        if (!real_fn) {
            // Runtime doesn't have _v2 — should not happen if NCCL calls it,
            // but fall back gracefully.
            fprintf(stderr, "[IPC-HOOK] WARNING: cuGetProcAddress_v2 not found in RTLD_NEXT\n");
            return CUDA_ERROR_NOT_FOUND;
        }
        fprintf(stderr, "[IPC-HOOK] cuGetProcAddress_v2 hook activated (5-param)\n");
    }
    CUresult ret = real_fn(symbol, pfn, cudaVersion, flags, symbolStatus);
    if (ret == CUDA_SUCCESS) cuGetProcAddress_common(symbol, pfn, "cuGetProcAddress_v2");
    return ret;
}

} // extern "C"

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
__attribute__((constructor))
static void ipc_hooks_init() {
    fprintf(stderr, "[IPC-HOOK] === ipc_hooks.cpp loaded (constructor) ===\n");

    void* our_gdep = dlsym(RTLD_DEFAULT, "cudaGetDriverEntryPoint");
    void* next_gdep = dlsym(RTLD_NEXT, "cudaGetDriverEntryPoint");
    Dl_info dl_info;
    fprintf(stderr, "[IPC-HOOK] cudaGetDriverEntryPoint:\n");
    if (our_gdep && dladdr(our_gdep, &dl_info))
        fprintf(stderr, "[IPC-HOOK]   DEFAULT -> %s\n", dl_info.dli_fname ? dl_info.dli_fname : "?");
    if (next_gdep && dladdr(next_gdep, &dl_info))
        fprintf(stderr, "[IPC-HOOK]   NEXT    -> %s\n", dl_info.dli_fname ? dl_info.dli_fname : "?");

    // Check both cuGetProcAddress (CUDA 11) and cuGetProcAddress_v2 (CUDA 12+)
    const char* gpa_names[] = { "cuGetProcAddress", "cuGetProcAddress_v2" };
    for (const char* name : gpa_names) {
        fprintf(stderr, "[IPC-HOOK] %s:\n", name);
        void* our_gpa = dlsym(RTLD_DEFAULT, name);
        void* next_gpa = dlsym(RTLD_NEXT, name);
        if (our_gpa && dladdr(our_gpa, &dl_info))
            fprintf(stderr, "[IPC-HOOK]   DEFAULT -> %s\n", dl_info.dli_fname ? dl_info.dli_fname : "?");
        if (next_gpa && dladdr(next_gpa, &dl_info))
            fprintf(stderr, "[IPC-HOOK]   NEXT    -> %s\n", dl_info.dli_fname ? dl_info.dli_fname : "?");
        if (!next_gpa)
            fprintf(stderr, "[IPC-HOOK]   NEXT    -> (nil)\n");
    }
}

// ---------------------------------------------------------------------------
// Public API: IPC event teardown
// ---------------------------------------------------------------------------

int ipc_teardown_all_events() {
    std::lock_guard<std::mutex> lock(g_ipc_events_mutex);
    if (g_ipc_events.empty()) return 0;

    typedef cudaError_t (*cudaEventDestroy_fn)(cudaEvent_t);
    static cudaEventDestroy_fn real_cudaEventDestroy = nullptr;
    if (!real_cudaEventDestroy) {
        real_cudaEventDestroy = (cudaEventDestroy_fn)dlsym(RTLD_NEXT, "cudaEventDestroy");
        if (!real_cudaEventDestroy) return -1;
    }

    int torn = 0;
    for (auto& rec : g_ipc_events) {
        if (rec.torn_down) continue;
        if (rec.is_opened) {
            cudaError_t ret = real_cudaEventDestroy(rec.event);
            if (ret != cudaSuccess)
                fprintf(stderr, "[IPC-HOOK] WARNING: cudaEventDestroy(%p) failed: %d\n",
                        (void*)rec.event, (int)ret);
        }
        rec.torn_down = true;
        torn++;
    }
    fprintf(stderr, "[IPC-HOOK] Teardown: %d IPC events processed\n", torn);
    return torn;
}

int ipc_get_event_count() {
    std::lock_guard<std::mutex> lock(g_ipc_events_mutex);
    int active = 0;
    for (const auto& rec : g_ipc_events) {
        if (!rec.torn_down) active++;
    }
    return active;
}

// ---------------------------------------------------------------------------
// Public API: Checkpoint — teardown imports
// ---------------------------------------------------------------------------

int ipc_teardown_all_imports() {
    std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
    resolve_helper_fns();
    auto t_total = std::chrono::high_resolution_clock::now();

    if (!real_cuMemUnmap)
        real_cuMemUnmap = (cuMemUnmap_fn)dlsym(RTLD_NEXT, "cuMemUnmap");
    if (!fn_cuMemRelease) {
        fprintf(stderr, "[IPC-HOOK] ERROR: cuMemRelease not resolved\n");
        return -1;
    }

    int torn = 0;
    size_t torn_bytes = 0;
    for (auto& rec : g_ipc_imports) {
        if (rec.torn_down) continue;

        fprintf(stderr, "[IPC-HOOK] Teardown import: vaddr=%p size=%zu handle=0x%llx\n",
                (void*)rec.mapped_vaddr, rec.size, (unsigned long long)rec.handle);

        if (real_cuMemUnmap) {
            CUresult res = real_cuMemUnmap(rec.mapped_vaddr, rec.size);
            if (res != CUDA_SUCCESS) {
                const char* e = "?"; cuGetErrorString(res, &e);
                fprintf(stderr, "[IPC-HOOK] WARNING: cuMemUnmap(%p) failed: %s\n",
                        (void*)rec.mapped_vaddr, e);
            }
        }

        if (rec.handle != 0) {
            CUresult res = fn_cuMemRelease(rec.handle);
            if (res != CUDA_SUCCESS) {
                const char* e = "?"; cuGetErrorString(res, &e);
                fprintf(stderr, "[IPC-HOOK] WARNING: cuMemRelease(0x%llx) failed: %s\n",
                        (unsigned long long)rec.handle, e);
            }
        } else {
            fprintf(stderr, "[IPC-HOOK] Import handle already released by application/library\n");
        }

        // NOTE: Do NOT call cuMemAddressFree — keep VA reserved so we can
        // reclaim the exact same address during import rebuild.

        rec.torn_down = true;
        rec.handle = 0;
        torn++;
        torn_bytes += rec.size;
    }
    auto t_total_end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_total_end - t_total).count();
    g_timing_snapshot.import_teardown_ms = total_ms;
    g_timing_snapshot.import_teardown_count = torn;
    g_timing_snapshot.import_teardown_bytes = torn_bytes;
    fprintf(stderr, "[IPC-HOOK] Teardown complete: %d IPC imports torn down\n", torn);
    return torn;
}

// ---------------------------------------------------------------------------
// Public API: Export-side checkpoint — save data + full teardown
// ---------------------------------------------------------------------------

size_t ipc_get_export_data_size() {
    std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
    size_t total = 0;
    for (const auto& rec : g_ipc_exports) {
        if (!rec.torn_down && rec.has_mapping && rec.mapped_size > 0) {
            total += rec.mapped_size;
        }
    }
    return total;
}

int ipc_save_and_teardown_all_exports(void* host_buf, size_t buf_size) {
    std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
    resolve_helper_fns();

    if (!real_cuMemUnmap)
        real_cuMemUnmap = (cuMemUnmap_fn)dlsym(RTLD_NEXT, "cuMemUnmap");

    if (!fn_cuMemRelease || !fn_cuMemcpyDtoH) {
        fprintf(stderr, "[IPC-HOOK] ERROR: required functions not resolved for export teardown\n");
        return -1;
    }

    auto t_step1 = std::chrono::high_resolution_clock::now();

    // Step 1: Save GPU data to host buffer — grouped by device, async batched
    // Group active exports by device ID to minimize context switches
    std::map<int, std::vector<size_t>> device_exports; // dev_id -> export indices
    for (size_t i = 0; i < g_ipc_exports.size(); i++) {
        auto& rec = g_ipc_exports[i];
        if (rec.torn_down || !rec.has_mapping || rec.mapped_size == 0) continue;
        int dev = rec.has_saved_prop ? rec.saved_prop.location.id : -1;
        device_exports[dev].push_back(i);
    }

    // Pre-compute total size to validate buffer
    size_t total_needed = 0;
    for (auto& [dev_id, indices] : device_exports) {
        for (size_t idx : indices) total_needed += g_ipc_exports[idx].mapped_size;
    }
    if (total_needed > buf_size) {
        fprintf(stderr, "[IPC-HOOK] ERROR: host buffer too small (need %zu, have %zu)\n",
                total_needed, buf_size);
        return -1;
    }

    // Try to pin host buffer for async transfer
    bool host_pinned = false;
    if (fn_cuMemHostRegister && total_needed > 0) {
        CUresult pr = fn_cuMemHostRegister(host_buf, total_needed, 0);
        if (pr == CUDA_SUCCESS) {
            host_pinned = true;
        } else {
            fprintf(stderr, "[IPC-HOOK] INFO: cuMemHostRegister failed (non-fatal), using sync DtoH\n");
        }
    }

    bool use_async = host_pinned && fn_cuStreamCreate && fn_cuStreamSynchronize &&
                     fn_cuStreamDestroy && fn_cuMemcpyDtoHAsync;

    size_t offset = 0;
    int count = 0;

    for (auto& [dev_id, indices] : device_exports) {
        // Push context once per device
        CUcontext dev_ctx = nullptr;
        bool ctx_pushed = false;
        if (dev_id >= 0 && fn_cuDevicePrimaryCtxRetain && fn_cuCtxPushCurrent && fn_cuCtxPopCurrent) {
            CUresult ctx_res = fn_cuDevicePrimaryCtxRetain(&dev_ctx, dev_id);
            if (ctx_res == CUDA_SUCCESS && dev_ctx) {
                fn_cuCtxPushCurrent(dev_ctx);
                ctx_pushed = true;
            }
        }

        if (use_async) {
            // Async path: batch all DtoH copies on a stream, synchronize once
            CUstream stream = nullptr;
            fn_cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);

            for (size_t idx : indices) {
                auto& rec = g_ipc_exports[idx];
                fprintf(stderr, "[IPC-HOOK] Async DtoH export: vaddr=%p size=%zu -> host+%zu (dev=%d)\n",
                        (void*)rec.mapped_vaddr, rec.mapped_size, offset, dev_id);
                rec.saved_data_offset = offset;
                rec.has_saved_data_offset = true;

                CUresult res = fn_cuMemcpyDtoHAsync((char*)host_buf + offset,
                                                     rec.mapped_vaddr, rec.mapped_size, stream);
                if (res != CUDA_SUCCESS) {
                    const char* e = "?"; cuGetErrorString(res, &e);
                    fprintf(stderr, "[IPC-HOOK] ERROR: cuMemcpyDtoHAsync(%p, %zu) failed: %s\n",
                            (void*)rec.mapped_vaddr, rec.mapped_size, e);
                    fn_cuStreamDestroy(stream);
                    if (ctx_pushed) {
                        CUcontext popped = nullptr;
                        fn_cuCtxPopCurrent(&popped);
                        fn_cuDevicePrimaryCtxRelease(dev_id);
                    }
                    if (host_pinned) fn_cuMemHostUnregister(host_buf);
                    return -1;
                }
                offset += rec.mapped_size;
                count++;
            }

            // Single sync for all copies on this device
            fn_cuStreamSynchronize(stream);
            fn_cuStreamDestroy(stream);
        } else {
            // Sync fallback: one context push per device, sequential copies
            for (size_t idx : indices) {
                auto& rec = g_ipc_exports[idx];
                fprintf(stderr, "[IPC-HOOK] Sync DtoH export: vaddr=%p size=%zu -> host+%zu (dev=%d)\n",
                        (void*)rec.mapped_vaddr, rec.mapped_size, offset, dev_id);
                rec.saved_data_offset = offset;
                rec.has_saved_data_offset = true;

                CUresult res = fn_cuMemcpyDtoH((char*)host_buf + offset,
                                                rec.mapped_vaddr, rec.mapped_size);
                if (res != CUDA_SUCCESS) {
                    const char* e = "?"; cuGetErrorString(res, &e);
                    fprintf(stderr, "[IPC-HOOK] ERROR: cuMemcpyDtoH(%p, %zu) failed: %s (dev=%d)\n",
                            (void*)rec.mapped_vaddr, rec.mapped_size, e, dev_id);
                    if (ctx_pushed) {
                        CUcontext popped = nullptr;
                        fn_cuCtxPopCurrent(&popped);
                        fn_cuDevicePrimaryCtxRelease(dev_id);
                    }
                    if (host_pinned) fn_cuMemHostUnregister(host_buf);
                    return -1;
                }
                offset += rec.mapped_size;
                count++;
            }
        }

        // Pop context once per device
        if (ctx_pushed) {
            CUcontext popped = nullptr;
            fn_cuCtxPopCurrent(&popped);
            if (fn_cuDevicePrimaryCtxRelease)
                fn_cuDevicePrimaryCtxRelease(dev_id);
        }
    }

    // Unpin host buffer
    if (host_pinned && fn_cuMemHostUnregister) {
        fn_cuMemHostUnregister(host_buf);
    }

    auto t_step1_end = std::chrono::high_resolution_clock::now();
    auto dtoh_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_step1_end - t_step1).count();
    fprintf(stderr, "[IPC-HOOK] Saved %d exports (%zu bytes) to host buffer (%s, %ld ms)\n",
            count, offset, use_async ? "async" : "sync", dtoh_ms);

    // Step 2: Close export fds
    auto t_step2 = std::chrono::high_resolution_clock::now();
    for (auto& rec : g_ipc_exports) {
        if (rec.torn_down) continue;
        if (rec.exported_fd >= 0) {
            close(rec.exported_fd);
            rec.exported_fd = -1;
        }
    }
    auto t_step2_end = std::chrono::high_resolution_clock::now();
    auto fd_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_step2_end - t_step2).count();
    fprintf(stderr, "[IPC-HOOK] Closed export fds (%ld ms)\n", fd_ms);

    // Step 3: Full teardown: cuMemUnmap + cuMemRelease (keep VA reserved)
    auto t_step3 = std::chrono::high_resolution_clock::now();
    int torn = 0;
    for (auto& rec : g_ipc_exports) {
        if (rec.torn_down) continue;
        if (!rec.has_mapping) {
            fprintf(stderr, "[IPC-HOOK] WARNING: export has no mapping, skipping teardown\n");
            continue;
        }

        // cuMemUnmap
        if (real_cuMemUnmap) {
            CUresult r = real_cuMemUnmap(rec.mapped_vaddr, rec.mapped_size);
            if (r != CUDA_SUCCESS) {
                const char* e = "?"; cuGetErrorString(r, &e);
                fprintf(stderr, "[IPC-HOOK] WARNING: cuMemUnmap export(%p) failed: %s\n",
                        (void*)rec.mapped_vaddr, e);
            }
        }

        // cuMemRelease
        CUresult r = fn_cuMemRelease(rec.local_handle);
        if (r != CUDA_SUCCESS) {
            const char* e = "?"; cuGetErrorString(r, &e);
            fprintf(stderr, "[IPC-HOOK] WARNING: cuMemRelease export(0x%llx) failed: %s\n",
                    (unsigned long long)rec.local_handle, e);
        }

        // NOTE: Do NOT call cuMemAddressFree — keep VA reserved so we can
        // reclaim the exact same address during rebuild.

        rec.torn_down = true;
        rec.local_handle = 0;
        torn++;
    }

    auto t_step3_end = std::chrono::high_resolution_clock::now();
    auto teardown_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_step3_end - t_step3).count();
    g_timing_snapshot.export_teardown_dtoh_ms = dtoh_ms;
    g_timing_snapshot.export_teardown_fd_close_ms = fd_ms;
    g_timing_snapshot.export_teardown_unmap_release_ms = teardown_ms;
    g_timing_snapshot.export_teardown_count = torn;
    g_timing_snapshot.export_teardown_bytes = offset;
    fprintf(stderr, "[IPC-HOOK] Export teardown complete: %d exports torn down (%ld ms)\n", torn, teardown_ms);
    fprintf(stderr, "[IPC-HOOK] === Export Teardown Timing: DtoH=%ld ms, FD-close=%ld ms, Unmap+Release=%ld ms ===\n",
            dtoh_ms, fd_ms, teardown_ms);
    return torn;
}

// ---------------------------------------------------------------------------
// Public API: Export-side restore — rebuild allocations + restore data + re-export
// ---------------------------------------------------------------------------

int ipc_rebuild_and_restore_all_exports(void* host_buf, size_t buf_size) {
    std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
    resolve_helper_fns();

    if (!real_cuMemCreate || !real_cuMemMap || !real_cuMemSetAccess ||
        !real_cuMemExportToShareableHandle || !fn_cuMemcpyHtoD || !fn_cuMemAddressReserve) {
        // Try fallback resolution
        if (!real_cuMemMap) real_cuMemMap = (cuMemMap_fn)dlsym(RTLD_DEFAULT, "cuMemMap");
        if (!real_cuMemSetAccess) real_cuMemSetAccess = (cuMemSetAccess_fn)dlsym(RTLD_DEFAULT, "cuMemSetAccess");
        if (!real_cuMemExportToShareableHandle)
            real_cuMemExportToShareableHandle = (cuMemExportToShareableHandle_fn)dlsym(RTLD_DEFAULT, "cuMemExportToShareableHandle");
    }

    if (!real_cuMemCreate || !real_cuMemMap || !fn_cuMemcpyHtoD || !fn_cuMemAddressReserve) {
        fprintf(stderr, "[IPC-HOOK] ERROR: required functions not resolved for export rebuild\n");
        return -1;
    }

    auto t_total = std::chrono::high_resolution_clock::now();

    // Phase A: cuMemCreate + cuMemMap + cuMemSetAccess for all exports (no data transfer yet)
    auto t_alloc = std::chrono::high_resolution_clock::now();

    // Collect exports to rebuild, grouped by device for batched HtoD later
    struct RebuildEntry {
        size_t export_idx;
        CUmemGenericAllocationHandle new_handle;
        size_t data_offset;
    };
    std::map<int, std::vector<RebuildEntry>> device_rebuilds; // dev_id -> entries

    int rebuilt = 0;
    size_t data_offset = 0;

    for (size_t i = 0; i < g_ipc_exports.size(); i++) {
        auto& rec = g_ipc_exports[i];
        if (!rec.torn_down) continue;
        if (!rec.has_mapping || rec.mapped_size == 0) continue;

        if (!rec.has_saved_prop) {
            fprintf(stderr, "[IPC-HOOK] ERROR: no saved prop for export vaddr=%p size=%zu\n",
                    (void*)rec.mapped_vaddr, rec.mapped_size);
            continue;
        }

        // Step 1: cuMemCreate
        CUmemGenericAllocationHandle new_handle;
        CUresult res = real_cuMemCreate(&new_handle, rec.mapped_size, &rec.saved_prop, 0);
        if (res != CUDA_SUCCESS) {
            const char* e = "?"; cuGetErrorString(res, &e);
            fprintf(stderr, "[IPC-HOOK] ERROR: cuMemCreate failed: %s\n", e);
            continue;
        }

        // Step 2: cuMemMap at original VA
        res = real_cuMemMap(rec.mapped_vaddr, rec.mapped_size, 0, new_handle, 0);
        if (res != CUDA_SUCCESS) {
            const char* e = "?"; cuGetErrorString(res, &e);
            fprintf(stderr, "[IPC-HOOK] ERROR: cuMemMap(%p) failed: %s\n",
                    (void*)rec.mapped_vaddr, e);
            fn_cuMemRelease(new_handle);
            continue;
        }

        // Step 3: cuMemSetAccess — replay tracked descriptors.
        // Tracking (hook_cuMemSetAccess) already dedups per (type, id), so we
        // replay the full vector directly. IMPORTANT: must NOT dedup by id
        // alone — NCCL with NCCL_CUMEM_HOST_ENABLE=1 sets both
        // (type=DEVICE, id=N) and (type=HOST_NUMA, id=N) on the same VA;
        // collapsing those by id would drop one access path and cause
        // cudaErrorIllegalAddress during NCCL collectives.
        {
            auto va_it = g_va_access_descs.find(rec.mapped_vaddr);
            if (va_it != g_va_access_descs.end() && !va_it->second.empty()) {
                int ok = 0, fail = 0;
                for (const auto& ad : va_it->second) {
                    CUresult ar = real_cuMemSetAccess(rec.mapped_vaddr, rec.mapped_size, &ad, 1);
                    if (ar != CUDA_SUCCESS) {
                        const char* e = "?"; cuGetErrorString(ar, &e);
                        fprintf(stderr, "[IPC-HOOK] WARNING: cuMemSetAccess(%p, type=%d id=%d) failed: %s\n",
                                (void*)rec.mapped_vaddr, (int)ad.location.type,
                                ad.location.id, e);
                        fail++;
                    } else {
                        ok++;
                    }
                }
                fprintf(stderr, "[IPC-HOOK] Replayed %zu access descriptors for export vaddr=%p (ok=%d, fail=%d)\n",
                        va_it->second.size(), (void*)rec.mapped_vaddr, ok, fail);
            } else if (rec.has_access) {
                real_cuMemSetAccess(rec.mapped_vaddr, rec.mapped_size, &rec.access_desc, 1);
            } else {
                CUmemAccessDesc access;
                memset(&access, 0, sizeof(access));
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = rec.saved_prop.location.id;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                real_cuMemSetAccess(rec.mapped_vaddr, rec.mapped_size, &access, 1);
            }
        }

        // Record for batched HtoD. Use the exact host-buffer offset recorded
        // during save; save groups copies by device and may not match vector
        // order on mixed-device export sets.
        int dev_id = rec.saved_prop.location.id;
        size_t restore_offset = rec.has_saved_data_offset ? rec.saved_data_offset : data_offset;
        device_rebuilds[dev_id].push_back({i, new_handle, restore_offset});
        data_offset = std::max(data_offset, restore_offset + rec.mapped_size);
        rebuilt++;
    }

    auto t_alloc_end = std::chrono::high_resolution_clock::now();
    auto alloc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_alloc_end - t_alloc).count();
    fprintf(stderr, "[IPC-HOOK] Alloc+Map+Access for %d exports (%ld ms)\n", rebuilt, alloc_ms);

    // Phase B: Batched HtoD data restore — grouped by device, async if possible
    auto t_htod = std::chrono::high_resolution_clock::now();

    // Try to pin host buffer for async transfer
    bool host_pinned = false;
    if (fn_cuMemHostRegister && data_offset > 0) {
        CUresult pr = fn_cuMemHostRegister(host_buf, data_offset, 0);
        if (pr == CUDA_SUCCESS) {
            host_pinned = true;
        }
    }
    bool use_async = host_pinned && fn_cuStreamCreate && fn_cuStreamSynchronize &&
                     fn_cuStreamDestroy && fn_cuMemcpyHtoDAsync;

    for (auto& [dev_id, entries] : device_rebuilds) {
        // Push context once per device
        CUcontext dev_ctx = nullptr;
        bool ctx_pushed = false;
        if (fn_cuDevicePrimaryCtxRetain && fn_cuCtxPushCurrent && fn_cuCtxPopCurrent) {
            CUresult ctx_res = fn_cuDevicePrimaryCtxRetain(&dev_ctx, dev_id);
            if (ctx_res == CUDA_SUCCESS && dev_ctx) {
                fn_cuCtxPushCurrent(dev_ctx);
                ctx_pushed = true;
            }
        }

        if (use_async) {
            CUstream stream = nullptr;
            fn_cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);

            for (auto& entry : entries) {
                auto& rec = g_ipc_exports[entry.export_idx];
                if (entry.data_offset + rec.mapped_size <= buf_size) {
                    fn_cuMemcpyHtoDAsync(rec.mapped_vaddr,
                                          (char*)host_buf + entry.data_offset,
                                          rec.mapped_size, stream);
                }
            }
            fn_cuStreamSynchronize(stream);
            fn_cuStreamDestroy(stream);
        } else {
            for (auto& entry : entries) {
                auto& rec = g_ipc_exports[entry.export_idx];
                if (entry.data_offset + rec.mapped_size <= buf_size) {
                    CUresult res = fn_cuMemcpyHtoD(rec.mapped_vaddr,
                                                    (char*)host_buf + entry.data_offset,
                                                    rec.mapped_size);
                    if (res != CUDA_SUCCESS) {
                        const char* e = "?"; cuGetErrorString(res, &e);
                        fprintf(stderr, "[IPC-HOOK] ERROR: cuMemcpyHtoD(%p, %zu) failed: %s\n",
                                (void*)rec.mapped_vaddr, rec.mapped_size, e);
                    }
                }
            }
        }

        if (ctx_pushed) {
            CUcontext popped = nullptr;
            fn_cuCtxPopCurrent(&popped);
            if (fn_cuDevicePrimaryCtxRelease) fn_cuDevicePrimaryCtxRelease(dev_id);
        }
    }

    if (host_pinned && fn_cuMemHostUnregister) fn_cuMemHostUnregister(host_buf);

    auto t_htod_end = std::chrono::high_resolution_clock::now();
    auto htod_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_htod_end - t_htod).count();
    fprintf(stderr, "[IPC-HOOK] HtoD data restore for %d exports (%s, %ld ms)\n",
            rebuilt, use_async ? "async" : "sync", htod_ms);

    // Phase C: Re-export to get new shareable fds
    auto t_export = std::chrono::high_resolution_clock::now();

    for (auto& [dev_id, entries] : device_rebuilds) {
        for (auto& entry : entries) {
            auto& rec = g_ipc_exports[entry.export_idx];
            int new_fd = -1;
            if (real_cuMemExportToShareableHandle) {
                CUresult res = real_cuMemExportToShareableHandle(&new_fd, entry.new_handle,
                                                                  rec.handle_type, 0);
                if (res != CUDA_SUCCESS) {
                    const char* e = "?"; cuGetErrorString(res, &e);
                    fprintf(stderr, "[IPC-HOOK] ERROR: re-export failed: %s\n", e);
                    new_fd = -1;
                }
            }

            if (rec.app_local_handle != 0 && rec.app_local_handle != entry.new_handle) {
                g_rebuilt_handle_alias[rec.app_local_handle] = entry.new_handle;
                fprintf(stderr, "[IPC-HOOK] Export handle alias app=0x%llx -> rebuilt=0x%llx\n",
                        (unsigned long long)rec.app_local_handle,
                        (unsigned long long)entry.new_handle);
            }
            rec.local_handle = entry.new_handle;
            rec.exported_fd = new_fd;
            rec.torn_down = false;

            CuMemCreateRecord new_create_rec;
            new_create_rec.size = rec.mapped_size;
            new_create_rec.prop = rec.saved_prop;
            g_created_allocs[entry.new_handle] = new_create_rec;
            g_handle_to_va[entry.new_handle] = rec.mapped_vaddr;
        }
    }

    auto t_export_end = std::chrono::high_resolution_clock::now();
    auto export_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_export_end - t_export).count();

    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_export_end - t_total).count();
    g_timing_snapshot.export_rebuild_alloc_map_access_ms = alloc_ms;
    g_timing_snapshot.export_rebuild_htod_ms = htod_ms;
    g_timing_snapshot.export_rebuild_reexport_ms = export_ms;
    g_timing_snapshot.export_rebuild_total_ms = total_ms;
    g_timing_snapshot.export_rebuild_count = rebuilt;
    g_timing_snapshot.export_rebuild_bytes = data_offset;
    fprintf(stderr, "[IPC-HOOK] Export rebuild complete: %d exports rebuilt (%zu bytes restored)\n",
            rebuilt, data_offset);
    fprintf(stderr, "[IPC-HOOK] === Export Rebuild Timing: Alloc+Map+Access=%ld ms, HtoD=%ld ms, Re-export=%ld ms, Total=%ld ms ===\n",
            alloc_ms, htod_ms, export_ms, total_ms);
    return rebuilt;
}

// ---------------------------------------------------------------------------
// Non-exported cuMem allocs: teardown + rebuild
// These are cuMemCreate allocations with handleTypes!=0 (IPC-capable) that
// NCCL uses internally but never exports. cuda-checkpoint cannot restore
// them because they have IPC driver state. We must teardown before checkpoint
// and rebuild after restore.
// ---------------------------------------------------------------------------

size_t ipc_get_local_alloc_data_size() {
    std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
    size_t total = 0;

    // Build set of exported VAs to exclude (handle may be zeroed after teardown)
    std::set<CUdeviceptr> exported_vas;
    for (const auto& exp : g_ipc_exports) {
        if (exp.has_mapping) exported_vas.insert(exp.mapped_vaddr);
    }

    for (const auto& kv : g_created_allocs) {
        auto va_it = g_handle_to_va.find(kv.first);
        if (va_it == g_handle_to_va.end()) continue;
        if (exported_vas.count(va_it->second)) continue;  // skip exported ones
        total += kv.second.size;
    }
    return total;
}

int ipc_save_and_teardown_local_allocs(void* host_buf, size_t buf_size) {
    std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
    resolve_helper_fns();

    if (!real_cuMemUnmap)
        real_cuMemUnmap = (cuMemUnmap_fn)dlsym(RTLD_NEXT, "cuMemUnmap");
    if (!fn_cuMemRelease || !fn_cuMemcpyDtoH) {
        fprintf(stderr, "[IPC-HOOK] ERROR: required functions not resolved for local alloc teardown\n");
        return -1;
    }

    // Build set of exported VAs to exclude (handle may be zeroed after teardown)
    std::set<CUdeviceptr> exported_vas;
    for (const auto& exp : g_ipc_exports) {
        if (exp.has_mapping) exported_vas.insert(exp.mapped_vaddr);
    }

    // Phase 1: Collect local allocs and group by device
    g_local_allocs.clear();

    struct LocalAllocEntry {
        CUmemGenericAllocationHandle handle;
        CUdeviceptr vaddr;
        size_t size;
        CUmemAllocationProp prop;
        size_t offset;
    };
    std::map<int, std::vector<LocalAllocEntry>> device_allocs; // dev_id -> entries
    size_t offset = 0;

    for (const auto& kv : g_created_allocs) {
        auto va_it = g_handle_to_va.find(kv.first);
        if (va_it == g_handle_to_va.end()) continue;
        if (exported_vas.count(va_it->second)) continue;

        size_t size = kv.second.size;
        if (offset + size > buf_size) {
            fprintf(stderr, "[IPC-HOOK] ERROR: local alloc buffer too small\n");
            return -1;
        }

        int dev_id = kv.second.prop.location.id;
        device_allocs[dev_id].push_back({kv.first, va_it->second, size, kv.second.prop, offset});
        offset += size;
    }

    // Phase 2: DtoH grouped by device with async batching
    auto t_dtoh = std::chrono::high_resolution_clock::now();

    // Try to pin host buffer
    bool host_pinned = false;
    if (fn_cuMemHostRegister && offset > 0) {
        CUresult pr = fn_cuMemHostRegister(host_buf, offset, 0);
        if (pr == CUDA_SUCCESS) host_pinned = true;
    }
    bool use_async = host_pinned && fn_cuStreamCreate && fn_cuStreamSynchronize &&
                     fn_cuStreamDestroy && fn_cuMemcpyDtoHAsync;

    int count = 0;
    for (auto& [dev_id, entries] : device_allocs) {
        CUcontext dev_ctx = nullptr;
        bool ctx_pushed = false;
        if (fn_cuDevicePrimaryCtxRetain && fn_cuCtxPushCurrent && fn_cuCtxPopCurrent) {
            CUresult cr = fn_cuDevicePrimaryCtxRetain(&dev_ctx, dev_id);
            if (cr == CUDA_SUCCESS && dev_ctx) {
                fn_cuCtxPushCurrent(dev_ctx);
                ctx_pushed = true;
            }
        }

        if (use_async) {
            CUstream stream = nullptr;
            fn_cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
            for (auto& e : entries) {
                CUresult res = fn_cuMemcpyDtoHAsync((char*)host_buf + e.offset, e.vaddr, e.size, stream);
                if (res != CUDA_SUCCESS) {
                    const char* err = "?"; cuGetErrorString(res, &err);
                    fprintf(stderr, "[IPC-HOOK] ERROR: async DtoH local alloc vaddr=%p: %s\n",
                            (void*)e.vaddr, err);
                }
            }
            fn_cuStreamSynchronize(stream);
            fn_cuStreamDestroy(stream);
        } else {
            for (auto& e : entries) {
                CUresult res = fn_cuMemcpyDtoH((char*)host_buf + e.offset, e.vaddr, e.size);
                if (res != CUDA_SUCCESS) {
                    const char* err = "?"; cuGetErrorString(res, &err);
                    fprintf(stderr, "[IPC-HOOK] ERROR: DtoH local alloc vaddr=%p size=%zu: %s\n",
                            (void*)e.vaddr, e.size, err);
                    if (ctx_pushed) { CUcontext p; fn_cuCtxPopCurrent(&p); fn_cuDevicePrimaryCtxRelease(dev_id); }
                    if (host_pinned) fn_cuMemHostUnregister(host_buf);
                    return -1;
                }
            }
        }

        if (ctx_pushed) {
            CUcontext popped = nullptr;
            fn_cuCtxPopCurrent(&popped);
            if (fn_cuDevicePrimaryCtxRelease) fn_cuDevicePrimaryCtxRelease(dev_id);
        }

        // Build g_local_allocs records
        for (auto& e : entries) {
            CuMemLocalAllocRecord rec;
            rec.handle = e.handle;
            rec.vaddr = e.vaddr;
            rec.size = e.size;
            rec.prop = e.prop;
            rec.torn_down = false;
            g_local_allocs.push_back(rec);
            count++;
        }
    }

    if (host_pinned && fn_cuMemHostUnregister) fn_cuMemHostUnregister(host_buf);

    auto t_dtoh_end = std::chrono::high_resolution_clock::now();
    auto dtoh_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_dtoh_end - t_dtoh).count();
    fprintf(stderr, "[IPC-HOOK] Saved %d local cuMem allocs (%zu bytes, %s, %ld ms)\n",
            count, offset, use_async ? "async" : "sync", dtoh_ms);

    // Phase 3: Teardown: cuMemUnmap + cuMemRelease (keep VA reserved)
    auto t_tear = std::chrono::high_resolution_clock::now();
    int torn = 0;
    for (auto& rec : g_local_allocs) {
        if (real_cuMemUnmap) {
            real_cuMemUnmap(rec.vaddr, rec.size);
        }
        fn_cuMemRelease(rec.handle);
        rec.torn_down = true;
        torn++;
    }

    auto t_tear_end = std::chrono::high_resolution_clock::now();
    auto tear_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_tear_end - t_tear).count();
    g_timing_snapshot.local_teardown_dtoh_ms = dtoh_ms;
    g_timing_snapshot.local_teardown_unmap_release_ms = tear_ms;
    g_timing_snapshot.local_teardown_count = torn;
    g_timing_snapshot.local_teardown_bytes = offset;
    fprintf(stderr, "[IPC-HOOK] Torn down %d local cuMem allocs (%ld ms)\n", torn, tear_ms);
    fprintf(stderr, "[IPC-HOOK] === Local Alloc Teardown Timing: DtoH=%ld ms, Unmap+Release=%ld ms ===\n",
            dtoh_ms, tear_ms);
    return torn;
}

int ipc_rebuild_local_allocs(void* host_buf, size_t buf_size) {
    std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
    resolve_helper_fns();

    if (!real_cuMemCreate || !real_cuMemMap || !fn_cuMemcpyHtoD) {
        fprintf(stderr, "[IPC-HOOK] ERROR: required functions not resolved for local alloc rebuild\n");
        return -1;
    }

    auto t_total = std::chrono::high_resolution_clock::now();

    // Phase A: cuMemCreate + cuMemMap + cuMemSetAccess (no data transfer)
    auto t_alloc = std::chrono::high_resolution_clock::now();

    struct LocalRebuildEntry {
        size_t alloc_idx;
        CUmemGenericAllocationHandle new_handle;
        size_t data_offset;
    };
    std::map<int, std::vector<LocalRebuildEntry>> device_rebuilds;

    int rebuilt = 0;
    size_t data_offset = 0;

    for (size_t i = 0; i < g_local_allocs.size(); i++) {
        auto& rec = g_local_allocs[i];
        if (!rec.torn_down) continue;

        CUmemGenericAllocationHandle new_handle;
        CUresult res = real_cuMemCreate(&new_handle, rec.size, &rec.prop, 0);
        if (res != CUDA_SUCCESS) {
            const char* e = "?"; cuGetErrorString(res, &e);
            fprintf(stderr, "[IPC-HOOK] ERROR: cuMemCreate local alloc: %s\n", e);
            continue;
        }

        res = real_cuMemMap(rec.vaddr, rec.size, 0, new_handle, 0);
        if (res != CUDA_SUCCESS) {
            const char* e = "?"; cuGetErrorString(res, &e);
            fprintf(stderr, "[IPC-HOOK] ERROR: cuMemMap local alloc vaddr=%p: %s\n",
                    (void*)rec.vaddr, e);
            fn_cuMemRelease(new_handle);
            continue;
        }

        // cuMemSetAccess — replay all tracked descriptors.
        // Tracking already dedups per (type, id); do NOT collapse by id alone
        // (DEVICE and HOST_NUMA can share the same id value but must both be
        // applied — required by NCCL_CUMEM_HOST_ENABLE=1).
        if (real_cuMemSetAccess) {
            auto va_it = g_va_access_descs.find(rec.vaddr);
            if (va_it != g_va_access_descs.end() && !va_it->second.empty()) {
                for (const auto& ad : va_it->second) {
                    real_cuMemSetAccess(rec.vaddr, rec.size, &ad, 1);
                }
            } else {
                CUmemAccessDesc access;
                memset(&access, 0, sizeof(access));
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = rec.prop.location.id;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                real_cuMemSetAccess(rec.vaddr, rec.size, &access, 1);
            }
        }

        int dev_id = rec.prop.location.id;
        device_rebuilds[dev_id].push_back({i, new_handle, data_offset});
        data_offset += rec.size;
        rebuilt++;
    }

    auto t_alloc_end = std::chrono::high_resolution_clock::now();
    auto alloc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_alloc_end - t_alloc).count();

    // Phase B: Batched HtoD — grouped by device, async if possible
    auto t_htod = std::chrono::high_resolution_clock::now();

    bool host_pinned = false;
    if (fn_cuMemHostRegister && data_offset > 0) {
        CUresult pr = fn_cuMemHostRegister(host_buf, data_offset, 0);
        if (pr == CUDA_SUCCESS) host_pinned = true;
    }
    bool use_async = host_pinned && fn_cuStreamCreate && fn_cuStreamSynchronize &&
                     fn_cuStreamDestroy && fn_cuMemcpyHtoDAsync;

    for (auto& [dev_id, entries] : device_rebuilds) {
        CUcontext dev_ctx = nullptr;
        bool ctx_pushed = false;
        if (fn_cuDevicePrimaryCtxRetain && fn_cuCtxPushCurrent && fn_cuCtxPopCurrent) {
            CUresult cr = fn_cuDevicePrimaryCtxRetain(&dev_ctx, dev_id);
            if (cr == CUDA_SUCCESS && dev_ctx) { fn_cuCtxPushCurrent(dev_ctx); ctx_pushed = true; }
        }

        if (use_async) {
            CUstream stream = nullptr;
            fn_cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
            for (auto& entry : entries) {
                auto& rec = g_local_allocs[entry.alloc_idx];
                if (entry.data_offset + rec.size <= buf_size) {
                    fn_cuMemcpyHtoDAsync(rec.vaddr, (char*)host_buf + entry.data_offset,
                                          rec.size, stream);
                }
            }
            fn_cuStreamSynchronize(stream);
            fn_cuStreamDestroy(stream);
        } else {
            for (auto& entry : entries) {
                auto& rec = g_local_allocs[entry.alloc_idx];
                if (entry.data_offset + rec.size <= buf_size) {
                    CUresult res = fn_cuMemcpyHtoD(rec.vaddr, (char*)host_buf + entry.data_offset, rec.size);
                    if (res != CUDA_SUCCESS) {
                        const char* e = "?"; cuGetErrorString(res, &e);
                        fprintf(stderr, "[IPC-HOOK] ERROR: HtoD local alloc vaddr=%p: %s\n",
                                (void*)rec.vaddr, e);
                    }
                }
            }
        }

        if (ctx_pushed) {
            CUcontext popped = nullptr;
            fn_cuCtxPopCurrent(&popped);
            if (fn_cuDevicePrimaryCtxRelease) fn_cuDevicePrimaryCtxRelease(dev_id);
        }

        // Update tracking
        for (auto& entry : entries) {
            auto& rec = g_local_allocs[entry.alloc_idx];
            CuMemCreateRecord cr_rec;
            cr_rec.size = rec.size;
            cr_rec.prop = rec.prop;
            g_created_allocs[entry.new_handle] = cr_rec;
            g_handle_to_va[entry.new_handle] = rec.vaddr;
            g_created_allocs.erase(rec.handle);
            g_handle_to_va.erase(rec.handle);
            rec.handle = entry.new_handle;
            rec.torn_down = false;
        }
    }

    if (host_pinned && fn_cuMemHostUnregister) fn_cuMemHostUnregister(host_buf);

    auto t_htod_end = std::chrono::high_resolution_clock::now();
    auto htod_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_htod_end - t_htod).count();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_htod_end - t_total).count();

    g_timing_snapshot.local_rebuild_alloc_map_access_ms = alloc_ms;
    g_timing_snapshot.local_rebuild_htod_ms = htod_ms;
    g_timing_snapshot.local_rebuild_total_ms = total_ms;
    g_timing_snapshot.local_rebuild_count = rebuilt;
    g_timing_snapshot.local_rebuild_bytes = data_offset;
    fprintf(stderr, "[IPC-HOOK] Rebuilt %d local cuMem allocs (%zu bytes restored)\n", rebuilt, data_offset);
    fprintf(stderr, "[IPC-HOOK] === Local Alloc Rebuild Timing: Alloc+Map+Access=%ld ms, HtoD=%ld ms, Total=%ld ms ===\n",
            alloc_ms, htod_ms, total_ms);
    return rebuilt;
}

// ---------------------------------------------------------------------------

int ipc_write_export_info_to_shm(IpcRebuildShmBlock* block) {
    std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
    if (!block) return -1;

    block->num_exports = 0;
    pid_t my_pid = getpid();

    for (size_t i = 0; i < g_ipc_exports.size() && i < IPC_MAX_EXPORTS_PER_PROC; i++) {
        const auto& rec = g_ipc_exports[i];
        if (rec.exported_fd < 0) continue;

        IpcExportShmEntry& entry = block->entries[block->num_exports];
        entry.owner_pid = my_pid;
        entry.fd = rec.exported_fd;
        entry.local_ptr = (void*)rec.mapped_vaddr;
        entry.size = rec.size;
        entry.valid = 1;
        block->num_exports++;
    }
    fprintf(stderr, "[IPC-HOOK] Wrote %d export entries to shm (pid=%d)\n",
            block->num_exports, my_pid);
    return block->num_exports;
}

int ipc_import_from_shm_block(const IpcRebuildShmBlock* block) {
    std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
    auto t_total = std::chrono::high_resolution_clock::now();
    if (!block || block->num_exports == 0) {
        g_timing_snapshot.import_rebuild_ms = 0;
        g_timing_snapshot.import_rebuild_count = 0;
        g_timing_snapshot.import_rebuild_bytes = 0;
        return 0;
    }

    if (!real_cuMemImportFromShareableHandle || !real_cuMemMap || !real_cuMemSetAccess) {
        // Fallback resolution
        if (!real_cuMemImportFromShareableHandle)
            real_cuMemImportFromShareableHandle = (cuMemImportFromShareableHandle_fn)dlsym(RTLD_DEFAULT, "cuMemImportFromShareableHandle");
        if (!real_cuMemMap)
            real_cuMemMap = (cuMemMap_fn)dlsym(RTLD_DEFAULT, "cuMemMap");
        if (!real_cuMemSetAccess)
            real_cuMemSetAccess = (cuMemSetAccess_fn)dlsym(RTLD_DEFAULT, "cuMemSetAccess");
    }

    if (!real_cuMemImportFromShareableHandle || !real_cuMemMap) {
        fprintf(stderr, "[IPC-HOOK] ERROR: cuMem functions not available for import rebuild\n");
        return -1;
    }

    // --- UDS-based fd transfer: group exports by owner_pid ---
    // Collect all peer fds grouped by owner PID so we can batch-request
    // them in a single UDS connection per peer.
    struct PeerFdRequest {
        pid_t pid;
        std::vector<int> peer_fds;       // fd numbers in the peer process
        std::vector<int> export_indices;  // corresponding index into block->entries[]
    };
    std::map<pid_t, PeerFdRequest> peer_requests;

    for (int i = 0; i < block->num_exports; i++) {
        const IpcExportShmEntry& shm_entry = block->entries[i];
        if (!shm_entry.valid) continue;

        auto& req = peer_requests[shm_entry.owner_pid];
        req.pid = shm_entry.owner_pid;
        req.peer_fds.push_back(shm_entry.fd);
        req.export_indices.push_back(i);
    }

    // For each peer, request all fds in one UDS connection
    // Map: (block entry index) -> local fd received via SCM_RIGHTS
    std::map<int, int> entry_to_local_fd;

    for (auto& kv : peer_requests) {
        PeerFdRequest& req = kv.second;
        int num = (int)req.peer_fds.size();

        fprintf(stderr, "[IPC-HOOK] Requesting %d fds from peer pid=%d via UDS\n",
                num, req.pid);

        std::vector<int> local_fds(num, -1);
        int received = uds_receive_fds(req.pid, req.peer_fds.data(), num, local_fds.data());

        if (received <= 0) {
            fprintf(stderr, "[IPC-HOOK] ERROR: uds_receive_fds(pid=%d) failed (received=%d)\n",
                    req.pid, received);
            continue;
        }

        for (int j = 0; j < received && j < num; j++) {
            entry_to_local_fd[req.export_indices[j]] = local_fds[j];
        }

        fprintf(stderr, "[IPC-HOOK] Got %d/%d fds from peer pid=%d\n",
                received, num, req.pid);
    }

    // -----------------------------------------------------------------------
    // Positional matching: NCCL creates exports/imports in deterministic
    // channel order.  The Nth export from a peer corresponds to the Nth
    // import we originally received from that peer.  Group peer_block
    // entries by owner_pid and match them positionally to our torn-down
    // imports (also in creation order).  Size-based matching is wrong when
    // multiple allocations share the same size (the common case for NCCL).
    // -----------------------------------------------------------------------

    // Collect torn-down imports in their original creation order
    struct TornImport {
        size_t idx;               // index in g_ipc_imports
        IpcImportRecord* rec;
    };
    std::vector<TornImport> torn_imports;
    for (size_t k = 0; k < g_ipc_imports.size(); k++) {
        if (g_ipc_imports[k].torn_down) {
            torn_imports.push_back({k, &g_ipc_imports[k]});
        }
    }

    // Collect valid export entries that we have local fds for, in block order
    struct ValidExport {
        int block_idx;
        int local_fd;
        const IpcExportShmEntry* entry;
    };
    std::vector<ValidExport> valid_exports;
    for (int i = 0; i < block->num_exports; i++) {
        const IpcExportShmEntry& shm_entry = block->entries[i];
        if (!shm_entry.valid) continue;

        auto fd_it = entry_to_local_fd.find(i);
        if (fd_it == entry_to_local_fd.end() || fd_it->second < 0) {
            fprintf(stderr, "[IPC-HOOK] ERROR: no local fd for export entry %d (pid=%d, fd=%d)\n",
                    i, shm_entry.owner_pid, shm_entry.fd);
            continue;
        }
        valid_exports.push_back({i, fd_it->second, &shm_entry});
    }

    fprintf(stderr, "[IPC-HOOK] Positional matching: %zu torn-down imports, %zu valid peer exports\n",
            torn_imports.size(), valid_exports.size());

    if (valid_exports.size() != torn_imports.size()) {
        fprintf(stderr, "[IPC-HOOK] WARNING: count mismatch — exports=%zu imports=%zu, "
                "will match min(%zu,%zu) positionally\n",
                valid_exports.size(), torn_imports.size(),
                valid_exports.size(), torn_imports.size());
    }

    int rebuilt = 0;
    size_t rebuilt_bytes = 0;
    size_t match_count = std::min(valid_exports.size(), torn_imports.size());

    for (size_t m = 0; m < match_count; m++) {
        auto& ve = valid_exports[m];
        IpcImportRecord* target = torn_imports[m].rec;

        fprintf(stderr, "[IPC-HOOK] Match[%zu]: peer_export(pid=%d, ptr=%p, size=%zu) "
                "-> import(vaddr=%p, size=%zu)\n",
                m, ve.entry->owner_pid, ve.entry->local_ptr, ve.entry->size,
                (void*)target->mapped_vaddr, target->size);

        CUmemGenericAllocationHandle new_handle;
        CUresult res = real_cuMemImportFromShareableHandle(
            &new_handle, (void*)(uintptr_t)ve.local_fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
        close(ve.local_fd);

        if (res != CUDA_SUCCESS) {
            const char* e = "?"; cuGetErrorString(res, &e);
            fprintf(stderr, "[IPC-HOOK] ERROR: cuMemImportFromShareableHandle(fd=%d) failed: %s\n",
                    ve.local_fd, e);
            continue;
        }

        // VA reservation was kept during teardown, no need to re-reserve.

        res = real_cuMemMap(target->mapped_vaddr, target->size, 0, new_handle, 0);
        if (res != CUDA_SUCCESS) {
            const char* e = "?"; cuGetErrorString(res, &e);
            fprintf(stderr, "[IPC-HOOK] ERROR: cuMemMap(%p) failed: %s\n",
                    (void*)target->mapped_vaddr, e);
            fn_cuMemRelease(new_handle);
            continue;
        }

        // Replay ALL access descriptors (not just the last one)
        if (!target->all_access.empty() && real_cuMemSetAccess) {
            int access_ok = 0, access_fail = 0;
            for (const auto& ad : target->all_access) {
                CUresult ar = real_cuMemSetAccess(target->mapped_vaddr, target->size, &ad, 1);
                if (ar != CUDA_SUCCESS) {
                    const char* e = "?"; cuGetErrorString(ar, &e);
                    fprintf(stderr, "[IPC-HOOK] WARNING: cuMemSetAccess(%p, dev=%d) failed: %s\n",
                            (void*)target->mapped_vaddr, ad.location.id, e);
                    access_fail++;
                } else {
                    access_ok++;
                }
            }
            fprintf(stderr, "[IPC-HOOK] Replayed %d/%zu access descriptors for import vaddr=%p "
                    "(ok=%d, fail=%d)\n",
                    access_ok + access_fail, target->all_access.size(),
                    (void*)target->mapped_vaddr, access_ok, access_fail);
        } else if (target->access_dev >= 0 && real_cuMemSetAccess) {
            // Fallback: old single-descriptor path
            real_cuMemSetAccess(target->mapped_vaddr, target->size, &target->access_desc, 1);
            fprintf(stderr, "[IPC-HOOK] Fallback: single access descriptor for import vaddr=%p dev=%d\n",
                    (void*)target->mapped_vaddr, target->access_dev);
        } else {
            fprintf(stderr, "[IPC-HOOK] WARNING: no access descriptors to replay for import vaddr=%p\n",
                    (void*)target->mapped_vaddr);
        }

        target->handle = new_handle;
        target->torn_down = false;
        rebuilt++;
        rebuilt_bytes += target->size;
        fprintf(stderr, "[IPC-HOOK] Rebuilt import[%zu]: vaddr=%p size=%zu new_handle=0x%llx\n",
                m, (void*)target->mapped_vaddr, target->size, (unsigned long long)new_handle);
    }

    // Close any unmatched fds
    for (size_t m = match_count; m < valid_exports.size(); m++) {
        fprintf(stderr, "[IPC-HOOK] WARNING: unmatched peer export %d (pid=%d) — closing fd\n",
                valid_exports[m].block_idx, valid_exports[m].entry->owner_pid);
        close(valid_exports[m].local_fd);
    }

    auto t_total_end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_total_end - t_total).count();
    g_timing_snapshot.import_rebuild_ms = total_ms;
    g_timing_snapshot.import_rebuild_count = rebuilt;
    g_timing_snapshot.import_rebuild_bytes = rebuilt_bytes;
    fprintf(stderr, "[IPC-HOOK] Rebuilt %d/%zu imports from peers (positional matching)\n",
            rebuilt, torn_imports.size());
    return rebuilt;
}

// ---------------------------------------------------------------------------
// State query
// ---------------------------------------------------------------------------

int ipc_get_import_count() {
    std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
    return (int)g_ipc_imports.size();
}

int ipc_get_export_count() {
    std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
    return (int)g_ipc_exports.size();
}

void ipc_dump_state() {
    std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
    fprintf(stderr, "\n========== IPC Hook State ==========\n");
    fprintf(stderr, "Imports (%zu):\n", g_ipc_imports.size());
    for (size_t i = 0; i < g_ipc_imports.size(); i++) {
        const auto& r = g_ipc_imports[i];
        fprintf(stderr, "  [%zu] vaddr=%p size=%zu handle=0x%llx dev=%d torn=%d\n",
                i, (void*)r.mapped_vaddr, r.size, (unsigned long long)r.handle,
                r.access_dev, r.torn_down);
    }
    fprintf(stderr, "Exports (%zu):\n", g_ipc_exports.size());
    for (size_t i = 0; i < g_ipc_exports.size(); i++) {
        const auto& r = g_ipc_exports[i];
        fprintf(stderr, "  [%zu] handle=0x%llx fd=%d vaddr=%p size=%zu torn=%d\n",
                i, (unsigned long long)r.local_handle, r.exported_fd,
                (void*)r.mapped_vaddr, r.size, r.torn_down);
    }
    fprintf(stderr, "Created allocs tracked: %zu\n", g_created_allocs.size());
    fprintf(stderr, "====================================\n\n");
}

void ipc_dump_nvidia_fds(const char* label) {
    char path[256], target[512];
    int nvidia_count = 0;
    fprintf(stderr, "\n[IPC-DIAG] === FD Scan: %s (pid=%d) ===\n", label, getpid());
    for (int fd = 0; fd < 1024; fd++) {
        snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
        ssize_t len = readlink(path, target, sizeof(target) - 1);
        if (len <= 0) continue;
        target[len] = '\0';
        if (strstr(target, "nvidia") || strstr(target, "cuda") ||
            strstr(target, "anon_inode") || strstr(target, "/dev/nv")) {
            fprintf(stderr, "[IPC-DIAG]   fd=%d -> %s\n", fd, target);
            nvidia_count++;
        }
    }
    fprintf(stderr, "[IPC-DIAG] Total NVIDIA/CUDA fds: %d\n", nvidia_count);
    fprintf(stderr, "[IPC-DIAG] ===========================\n\n");
}

int ipc_validate_all_mappings(const char* label) {
    std::lock_guard<std::recursive_mutex> lock(g_ipc_hook_mutex);
    resolve_helper_fns();

    fprintf(stderr, "\n[IPC-VALIDATE] === %s (pid=%d) ===\n", label, getpid());

    int errors = 0;
    int checked = 0;

    // Validate exports: try reading 4 bytes from each mapped VA
    for (size_t i = 0; i < g_ipc_exports.size(); i++) {
        const auto& rec = g_ipc_exports[i];
        if (rec.torn_down || !rec.has_mapping || rec.mapped_size == 0) continue;

        uint32_t probe = 0;
        CUcontext dev_ctx = nullptr;
        bool ctx_pushed = false;
        if (fn_cuDevicePrimaryCtxRetain && fn_cuCtxPushCurrent && fn_cuCtxPopCurrent) {
            CUresult cr = fn_cuDevicePrimaryCtxRetain(&dev_ctx, rec.saved_prop.location.id);
            if (cr == CUDA_SUCCESS && dev_ctx) {
                fn_cuCtxPushCurrent(dev_ctx);
                ctx_pushed = true;
            }
        }

        CUresult res = fn_cuMemcpyDtoH(&probe, rec.mapped_vaddr, sizeof(probe));

        if (ctx_pushed) {
            CUcontext popped = nullptr;
            fn_cuCtxPopCurrent(&popped);
            if (fn_cuDevicePrimaryCtxRelease)
                fn_cuDevicePrimaryCtxRelease(rec.saved_prop.location.id);
        }

        checked++;
        if (res != CUDA_SUCCESS) {
            const char* e = "?"; cuGetErrorString(res, &e);
            fprintf(stderr, "[IPC-VALIDATE] EXPORT[%zu] FAIL: vaddr=%p size=%zu dev=%d err=%s\n",
                    i, (void*)rec.mapped_vaddr, rec.mapped_size, rec.saved_prop.location.id, e);
            errors++;
        } else {
            fprintf(stderr, "[IPC-VALIDATE] EXPORT[%zu] OK: vaddr=%p size=%zu probe=0x%08x\n",
                    i, (void*)rec.mapped_vaddr, rec.mapped_size, probe);
        }
    }

    // Validate imports: try reading 4 bytes from each mapped VA
    for (size_t i = 0; i < g_ipc_imports.size(); i++) {
        const auto& rec = g_ipc_imports[i];
        if (rec.torn_down || rec.size == 0) continue;

        uint32_t probe = 0;
        CUresult res = fn_cuMemcpyDtoH(&probe, rec.mapped_vaddr, sizeof(probe));
        checked++;
        if (res != CUDA_SUCCESS) {
            const char* e = "?"; cuGetErrorString(res, &e);
            fprintf(stderr, "[IPC-VALIDATE] IMPORT[%zu] FAIL: vaddr=%p size=%zu dev=%d err=%s\n",
                    i, (void*)rec.mapped_vaddr, rec.size, rec.access_dev, e);
            errors++;
        } else {
            fprintf(stderr, "[IPC-VALIDATE] IMPORT[%zu] OK: vaddr=%p size=%zu probe=0x%08x\n",
                    i, (void*)rec.mapped_vaddr, rec.size, probe);
        }
    }

    // Validate local (non-exported) cuMem allocs
    for (size_t i = 0; i < g_local_allocs.size(); i++) {
        const auto& rec = g_local_allocs[i];
        if (rec.torn_down || rec.size == 0) continue;

        uint32_t probe = 0;
        CUcontext dev_ctx = nullptr;
        bool ctx_pushed = false;
        if (fn_cuDevicePrimaryCtxRetain && fn_cuCtxPushCurrent && fn_cuCtxPopCurrent) {
            CUresult cr = fn_cuDevicePrimaryCtxRetain(&dev_ctx, rec.prop.location.id);
            if (cr == CUDA_SUCCESS && dev_ctx) {
                fn_cuCtxPushCurrent(dev_ctx);
                ctx_pushed = true;
            }
        }

        CUresult res = fn_cuMemcpyDtoH(&probe, rec.vaddr, sizeof(probe));

        if (ctx_pushed) {
            CUcontext popped = nullptr;
            fn_cuCtxPopCurrent(&popped);
            if (fn_cuDevicePrimaryCtxRelease)
                fn_cuDevicePrimaryCtxRelease(rec.prop.location.id);
        }

        checked++;
        if (res != CUDA_SUCCESS) {
            const char* e = "?"; cuGetErrorString(res, &e);
            fprintf(stderr, "[IPC-VALIDATE] LOCAL[%zu] FAIL: vaddr=%p size=%zu dev=%d err=%s\n",
                    i, (void*)rec.vaddr, rec.size, rec.prop.location.id, e);
            errors++;
        } else {
            fprintf(stderr, "[IPC-VALIDATE] LOCAL[%zu] OK: vaddr=%p size=%zu probe=0x%08x\n",
                    i, (void*)rec.vaddr, rec.size, probe);
        }
    }

    fprintf(stderr, "[IPC-VALIDATE] Checked %d mappings, %d errors\n", checked, errors);
    fprintf(stderr, "[IPC-VALIDATE] ===========================\n\n");
    return errors;
}
